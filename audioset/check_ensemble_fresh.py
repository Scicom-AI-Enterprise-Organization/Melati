#!/usr/bin/env python3
"""Melati-AudioSet-Ensemble as someone else meets it: downloaded into an empty Hugging Face cache and loaded in a venv
that holds only what its card asks for, at the versions pip resolves today (deploy/box_ensemble_fresh.sh). Each
member must give the probabilities audioset/evaluate.py cached for it on the fixture's clips
(audioset/ensemble_fixture.py), and the card's Use snippet must run as written. Imports nothing from this repo, so it
runs in that venv. RUNS ON THE BOX.

    python audioset/check_ensemble_fresh.py --fixture $MELATI_DATA/tmp/ensemble_fixture.npz [--revision <commit>]
        [--no-snippet | --snippet-only] [--engine] [--out result.json]

With --engine, melati_engine.py too: its Engine on the fixture's clips (bf16), and its server started as the card
says, one clip posted to it as a WAV file, its answer the Engine's.
"""
import argparse
import importlib.metadata as md
import json
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO = "Scicom-intl/Melati-AudioSet-Ensemble"
PACKAGES = ("torch", "torchaudio", "transformers", "timm", "huggingface_hub", "safetensors", "numpy", "soundfile")


def versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def snippet(fx, revision):
    """The card's first python block, run in a fresh process on one fixture clip saved as clip.wav at 44.1 kHz (the
    snippet reads it back at 16 kHz) -> (its stdout, an error or None)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    import soundfile as sf
    import torch
    sys.path.insert(0, snapshot_download(REPO, revision=revision))
    from melati_ensemble import resample
    code = re.search(r"```python\n(.*?)```", Path(hf_hub_download(REPO, "README.md", revision=revision)).read_text(),
                     re.S).group(1)
    if revision:
        code = code.replace(f'snapshot_download("{REPO}")', f'snapshot_download("{REPO}", revision="{revision}")')
    x = torch.from_numpy(fx["wave"][0][:int(fx["lens"][0])].astype(np.float32))
    with tempfile.TemporaryDirectory() as d:
        sf.write(Path(d) / "clip.wav", resample(x, 16000, 44100).numpy(), 44100)
        (Path(d) / "use.py").write_text(code)
        r = subprocess.run([sys.executable, "use.py"], cwd=d, capture_output=True, text=True)
    return r.stdout, (r.stderr[-3000:] if r.returncode else None)


def engine(fx, path, revision):
    """melati_engine.py in this venv -> its result: the Engine on the fixture's clips against the cache (bf16, so the
    bound is looser than the members' fp32 one), and `melati_engine.py serve` on one clip posted as a WAV file."""
    import io
    import os
    import socket
    import urllib.request
    import soundfile as sf
    import melati_engine as me
    e = me.Engine(path, gpus=[0], max_batch=16)
    per = e.batcher.submit(fx["wave"]).result()
    out = {"load_s": e.load_s, "members": {}}
    for k, name in enumerate(e.names):
        d = np.abs(per[k] - fx[f"p_{name}"][:len(fx["wave"])])
        out["members"][name] = {"max_abs_diff": round(float(d.max()), 5), "mean_abs_diff": float(d.mean()),
                                "ok": bool(d.mean() < 2e-3 and (name == "ced-base" or d.max() < 0.1))}
        print(f"[fresh] engine {name}: max |p - cached| {d.max():.5f}, mean {d.mean():.2e}", flush=True)
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    cmd = [sys.executable, str(Path(path) / "melati_engine.py"), "serve", "--gpus", "0", "--port", str(port),
           "--host", "127.0.0.1", "--max-batch", "16"] + (["--revision", revision] if revision else [])
    server = subprocess.Popen(cmd, env={**os.environ}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                break
            except OSError:
                if server.poll() is not None or time.time() - t0 > 600:
                    server.terminate()
                    raise RuntimeError(f"the server did not come up: {server.communicate(timeout=60)[0][-3000:]}")
                time.sleep(2)
        x = fx["wave"][1][:int(fx["lens"][1])]
        buf = io.BytesIO()
        sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
        req = urllib.request.Request(f"http://127.0.0.1:{port}/tag?probs=true&top_k=5", data=buf.getvalue(),
                                     headers={"Content-Type": "audio/wav"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        _, p, _ = e.tag(me.read(io.BytesIO(buf.getvalue())))
        out["http_max_abs_diff"] = round(float(np.abs(np.array(r["probs"]) - p.max(0)).max()), 5)
        out["http_tags"] = [t["label"] for t in r["tags"]]
        print(f"[fresh] engine served {out['http_tags']}, against the Engine's own answer max |diff| "
              f"{out['http_max_abs_diff']}", flush=True)
    finally:
        server.terminate()
        server.wait(60)
    out["ok"] = all(m["ok"] for m in out["members"].values()) and out["http_max_abs_diff"] < 1e-3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n", type=int, default=0, help="clips to run (default: every clip in the fixture)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-snippet", action="store_true", help="skip the card's snippet (a README not yet updated)")
    ap.add_argument("--snippet-only", action="store_true", help="only the card's snippet")
    ap.add_argument("--engine", action="store_true", help="also melati_engine.py: the Engine, and its server")
    ap.add_argument("--out", type=Path, default=None, help="the result as JSON")
    a = ap.parse_args()
    res = {"repo": REPO, "revision": a.revision, "versions": versions()}
    print("[fresh] versions", json.dumps(res["versions"]), flush=True)
    fx = np.load(a.fixture)
    if not a.snippet_only:
        W = fx["wave"][:a.n or None]
        try:
            from huggingface_hub import snapshot_download
            t0 = time.time()
            path = snapshot_download(REPO, revision=a.revision)
            sys.path.insert(0, path)
            from melati_ensemble import Ensemble
            model = Ensemble(path, device=a.device)
            res["load_s"] = round(time.time() - t0, 1)
            res["device"] = str(model.device)
            print(f"[fresh] downloaded and loaded in {res['load_s']:.0f} s, on {model.device}", flush=True)
            t0 = time.time()
            avg, each = model(list(W), batch=16, each=True)
            res["clips"], res["run_s"] = len(W), round(time.time() - t0, 1)
        except Exception:                    # noqa: BLE001 - the failure IS the finding; report it whole
            res["error"] = traceback.format_exc()[-4000:]
            print(res["error"], flush=True)
            if a.out:
                a.out.write_text(json.dumps(res, indent=1))
            raise SystemExit("[fresh] FAILED to download, load or run")
        bad, res["members"] = [], {}
        for k, p in each.items():
            d = np.abs(p - fx[f"p_{k}"][:len(W)])
            # CED-base is given one clip at a time here and was cached in batches of 64, which moves its decibel floor
            ok = d.mean() < 1e-3 and (k == "ced-base" or d.max() < 0.01)
            res["members"][k] = {"max_abs_diff": round(float(d.max()), 5), "mean_abs_diff": float(d.mean()),
                                 "ok": bool(ok)}
            print(f"[fresh] {k}: max |p - cached| {d.max():.5f}, mean {d.mean():.2e} {'ok' if ok else 'OFF'}",
                  flush=True)
            bad += [] if ok else [k]
        ref = np.mean([fx[f"p_{k}"][:len(W)] for k in each], 0)
        res["ensemble_max_abs_diff"] = round(float(np.abs(avg - ref).max()), 5)
        if a.engine:
            del model
            res["engine"] = engine(fx, path, a.revision)
            bad += [] if res["engine"]["ok"] else ["the engine"]
        if a.out:
            a.out.write_text(json.dumps(res, indent=1))
        if bad:
            raise SystemExit(f"[fresh] off the cache: {bad}")
    if not a.no_snippet:
        out, err = snippet(fx, a.revision)
        print(f"[fresh] the card's snippet:\n{out}", flush=True)
        if err:
            print(err, flush=True)
            raise SystemExit("[fresh] the card's snippet FAILED")
    print("FRESH_OK", flush=True)


if __name__ == "__main__":
    main()
