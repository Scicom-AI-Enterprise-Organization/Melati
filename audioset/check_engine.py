#!/usr/bin/env python3
"""melati_engine.py (audioset/hub) measured: AudioSet's evaluation set through the engine (bf16 weights, CUDA graphs,
dynamic batching), whose mAP must be the release check's bf16 mAP (results/audioset/ensemble_fp32.json); its
throughput against Ensemble's own bf16 call, with and without CUDA graphs, on one GPU and on several; one clip's
latency; and, with --http, the HTTP server under concurrent requests, its answers the engine's own. With --revision,
the engine and the repo's files are the Hub's at that commit, loaded as a user would load them; with --path, the code
in audioset/hub over a local copy of the repo's files. Writes results/audioset/engine.json. RUNS ON THE BOX, in
.venv-sed.

    python audioset/check_engine.py (--revision <commit> | --path <repo folder>) --gpus 0,1,2,3 [--http]
"""
import os
import argparse
import gc
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from data import load  # noqa: E402
from evaluate import mean_ap  # noqa: E402
from release import RESULTS  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

OUT = RESULTS / "engine.json"
REPO = "Scicom-intl/Melati-AudioSet-Ensemble"
me = None                                          # melati_engine, imported in main() from where --revision says


def free():
    gc.collect()
    torch.cuda.empty_cache()


def through(engine, W, inflight=4096):
    """Every clip of W through the engine, at most `inflight` waiting -> ((members, clips, 527), seconds)."""
    out = np.empty((len(engine.names), len(W), 527), np.float32)
    sem = threading.Semaphore(inflight)

    def done(i):
        def cb(f):
            out[:, i] = f.result()[:, 0]
            sem.release()
        return cb
    t0 = time.perf_counter()
    futs = []
    for i in range(len(W)):
        sem.acquire()
        f = engine.batcher.submit(np.asarray(W[i], np.float32)[None])
        f.add_done_callback(done(i))
        futs.append(f)
    for f in futs:
        f.result()
    return out, time.perf_counter() - t0


def eager_rate(path, gpu, W, batch=64):
    """Ensemble's own call in bf16 autocast (no engine), clips a second over W."""
    ens = me.Ensemble(path, device=f"cuda:{gpu}", bf16=True)
    ens([np.asarray(w, np.float32) for w in W[:batch]], batch=batch)            # warm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(W), batch):
        ens([np.asarray(w, np.float32) for w in W[i:i + batch]], batch=batch)
    torch.cuda.synchronize()
    rate = len(W) / (time.perf_counter() - t0)
    del ens
    free()
    return round(rate, 1)


def wav_bytes(x):
    import soundfile as sf
    b = io.BytesIO()
    sf.write(b, x, 16000, format="WAV", subtype="PCM_16")
    return b.getvalue()


def http(path, gpus, W, port, expect, conc_levels=(1, 16, 64, 256), per_level=1024):
    """`melati_engine.py serve` started as the card says, in a process of its own on the same GPUs, then `conc`
    clients at once (asyncio, so the load is not the client's own) posting WAV files: answers a second and latency.
    expect: the in-process engine's answer for the first clip, which the server's must be."""
    import asyncio
    import os
    import subprocess
    import aiohttp
    import requests
    cmd = [sys.executable, me.__file__, "serve", "--gpus", gpus, "--port", str(port), "--host", "127.0.0.1",
           "--path", str(path)]
    log = open(f"{MELATI_DATA}/tmp/engine_serve.log", "w")       # the server's own output, for when it fails
    server = subprocess.Popen(cmd, env={**os.environ}, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while True:
            try:
                requests.get(f"{url}/health", timeout=5).raise_for_status()
                break
            except requests.RequestException:
                assert server.poll() is None and time.time() - t0 < 900, "the server did not come up"
                time.sleep(2)
        out = {"server_up_s": round(time.time() - t0, 1), "levels": []}
        bodies = [wav_bytes(np.asarray(W[i], np.float32)) for i in range(64)]
        r = requests.post(f"{url}/tag", data=bodies[0], params={"probs": "true"}, timeout=120).json()
        out["answer_max_abs_diff"] = round(float(np.abs(np.array(r["probs"]) - expect).max()), 5)
        long = requests.post(f"{url}/tag", data=wav_bytes(np.concatenate([np.asarray(W[i], np.float32)
                                                                          for i in range(3)])),
                             params={"windows": "true"}, timeout=120).json()
        out["long_30s_windows"] = long["windows"]
        print(f"[engine] http: up in {out['server_up_s']} s; answer vs the engine's max |diff| "
              f"{out['answer_max_abs_diff']}; 30 s -> {long['windows']} windows", flush=True)

        async def level(conc):
            lat, sem = [], asyncio.Semaphore(conc)
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=conc)) as session:
                async def one(i):
                    async with sem:
                        t1 = time.perf_counter()
                        async with session.post(f"{url}/tag", data=bodies[i % len(bodies)]) as resp:
                            await resp.read()
                            resp.raise_for_status()
                        lat.append((time.perf_counter() - t1) * 1000)
                t1 = time.perf_counter()
                await asyncio.gather(*[one(i) for i in range(per_level)])
                return per_level / (time.perf_counter() - t1), lat
        for conc in conc_levels:
            rate, lat = asyncio.run(level(conc))
            lvl = {"concurrency": conc, "requests": per_level, "req_per_s": round(rate, 1),
                   "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
                   "latency_ms_p95": round(float(np.percentile(lat, 95)), 1)}
            print(f"[engine] http {lvl}", flush=True)
            out["levels"].append(lvl)
        out["stats"] = requests.get(f"{url}/stats", timeout=10).json()
    finally:
        server.terminate()
        server.wait(120)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, help="a local copy of the repo's files, run with audioset/hub's code")
    ap.add_argument("--revision", default=None, help="the repo at this commit from the Hub, its own code")
    ap.add_argument("--gpus", default="0", help="the GPUs for the multi-GPU numbers; the first alone for the rest")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=18900)
    a = ap.parse_args()
    global me
    assert bool(a.path) != bool(a.revision), "--path or --revision"
    if a.revision:
        from huggingface_hub import snapshot_download
        a.path = snapshot_download(REPO, revision=a.revision)
    sys.path.insert(0, a.path if a.revision else str(HERE / "hub"))
    import melati_engine
    me = melati_engine
    gpus = [int(g) for g in a.gpus.split(",")]
    ref = json.loads((RESULTS / "ensemble_fp32.json").read_text())
    W, _, _, Y = load("eval")
    from importlib.metadata import version
    res = {"repo": REPO, "revision": a.revision, "clips": int(len(W)), "max_batch": a.max_batch,
           "versions": {p: version(p) for p in ("torch", "numpy", "huggingface_hub", "safetensors")}}
    sample = W[:2048]
    res["ensemble_bf16_b64_clips_per_s"] = eager_rate(a.path, gpus[0], sample)
    print(f"[engine] Ensemble(bf16=True), batches of 64: {res['ensemble_bf16_b64_clips_per_s']} clips/s", flush=True)
    for graphs in (False, True):
        e = me.Engine(a.path, gpus[:1], a.max_batch, graphs=graphs)
        _, s = through(e, sample)
        key = "graphs" if graphs else "eager"
        b = me.bench(e, clips=1024, latency_runs=30)
        res[f"one_gpu_{key}"] = {"clips_per_s": round(len(sample) / s, 1), "bench_clips_per_s": b["clips_per_s"],
                                 "latency_ms_p50": b["latency_ms_p50"], "latency_ms_p95": b["latency_ms_p95"]}
        print(f"[engine] one GPU, {key}: {res[f'one_gpu_{key}']}", flush=True)
        e.close()               # its graphs and cache, freed: a capture that runs short of memory fails
    e = me.Engine(a.path, gpus, a.max_batch)
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:                  # what one clip costs: matmuls, convolutions, attention
        e.replicas[0].ens.forward(e.replicas[0].x[:1])
    res["tflop_per_clip"] = round(fc.get_total_flops() / 1e12, 3)
    print(f"[engine] one clip: {res['tflop_per_clip']} TFLOP", flush=True)
    P, s = through(e, W)
    avg = P.mean(0)
    res["eval"] = {"gpus": gpus, "clips_per_s": round(len(W) / s, 1), "realtime": round(len(W) * 10 / s),
                   "map": round(mean_ap(avg, Y), 4), "release_map_bf16": ref["map_bf16"],
                   "members": {n: round(mean_ap(P[k], Y), 4) for k, n in enumerate(e.names)}}
    print(f"[engine] the evaluation set on GPUs {gpus}: {res['eval']}", flush=True)
    assert abs(res["eval"]["map"] - ref["map_bf16"]) <= 1e-4, (res["eval"]["map"], ref["map_bf16"])
    res["stats"] = e.batcher.report()
    if a.http:
        _, p, _ = e.tag(me.read(io.BytesIO(wav_bytes(np.asarray(W[0], np.float32)))))
        e.close()                                   # the GPUs, for the server's own process
        res["http"] = http(a.path, a.gpus, W, a.port, p.max(0))
    OUT.write_text(json.dumps(res, indent=1))
    print("ENGINE_OK", flush=True)


if __name__ == "__main__":
    main()
