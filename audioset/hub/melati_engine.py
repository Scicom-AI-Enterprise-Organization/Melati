"""An inference engine for Melati-AudioSet-Ensemble: every GPU holds all six members, their Linear and Conv2d weights
in bf16, and runs a CUDA graph captured for each batch size; the 10 s windows of every request share one queue, so
concurrent requests fill the same batches (dynamic batching) and every GPU takes its share. A recording longer than
10 s is cut into 10 s windows every hop_s seconds, the last ending where it ends; a clip's probability for a sound is
the highest over its windows. It gives melati_ensemble.Ensemble's bf16 probabilities (the card has the mAP).

    pip install torch numpy huggingface_hub safetensors soundfile fastapi uvicorn
    hf download Scicom-intl/Melati-AudioSet-Ensemble --local-dir melati-ensemble

    python melati-ensemble/melati_engine.py serve --gpus 0,1 --port 8000           # HTTP: POST audio to /tag
    python melati-ensemble/melati_engine.py tag clips/ --gpus 0,1 --out tags.jsonl  # every audio file under clips/
    python melati-ensemble/melati_engine.py bench --gpus 0                           # clips a second, one clip's latency

    curl -s --data-binary @clip.wav "localhost:8000/tag?top_k=5"

From Python: engine = Engine(gpus=[0]); starts, probs, each = engine.tag(wave_16k) (probs: (windows, 527)). It runs on
CUDA GPUs only; on a CPU, melati_ensemble.Ensemble does the same.
"""
import argparse
import asyncio
import base64
import collections
import io
import itertools
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(os.path.abspath(__file__)).parent      # not resolve(): in the Hub's cache this file is a symlink into
sys.path.insert(0, str(HERE))                      # blobs/, where melati_ensemble.py is not stored by its name
from melati_ensemble import CLIP, SR, Ensemble, read, resample  # noqa: E402

REPO = "Scicom-intl/Melati-AudioSet-Ensemble"
SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)          # the batch sizes a graph is captured for, up to max_batch
AUDIO = {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3"}


def cut(x, hop_s=5.0):
    """A recording (16 kHz mono) -> (each window's start in seconds, (windows, 160000) float32): 10 s windows every
    hop_s seconds, the last one ending where the recording ends; 10 s or less is one window, padded with zeros. The
    windows Ensemble.windows() takes."""
    x = np.asarray(x, np.float32)
    hop = int(round(hop_s * SR))
    starts = list(range(0, max(len(x) - CLIP, 0) + 1, hop))
    if len(x) > CLIP and starts[-1] + CLIP < len(x):
        starts.append(len(x) - CLIP)
    W = np.zeros((len(starts), CLIP), np.float32)
    for k, s in enumerate(starts):
        seg = x[s:s + CLIP]
        W[k, :len(seg)] = seg
    return np.array(starts) / SR, W


class Replica:
    """The six members on one GPU, their Linear and Conv2d weights in bf16 (Ensemble.bf16_weights), and a CUDA graph
    captured for each batch size in SIZES up to max_batch: a batch runs as the smallest graph it fits, its spare rows
    zeros (each clip's probabilities do not depend on the rest of its batch)."""

    def __init__(self, path, device, max_batch=64):
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.ens = Ensemble(path, device=self.device).bf16_weights()
            self.names = list(self.ens.members)
            self.sizes = [s for s in SIZES if s < max_batch] + [max_batch]
            self.x = torch.zeros(max_batch, CLIP, device=self.device)
            self.host = torch.zeros(max_batch, CLIP).pin_memory()
        self.graphs = {}

    def _forward(self, x):
        return torch.stack(list(self.ens.forward(x).values()))                 # (members, B, 527)

    def capture(self):
        """One graph per batch size, the largest first, all in one memory pool. Nothing else may use the GPUs while
        this runs (a capture refuses any other CUDA call), so the engine captures one replica at a time, each on a
        capture stream of its own GPU: torch.cuda.graph's default capture stream is made once, on whichever GPU
        captured first, and a later capture on another GPU would launch its kernels outside it."""
        with torch.cuda.device(self.device):
            pool, side, cap = torch.cuda.graph_pool_handle(), torch.cuda.Stream(), torch.cuda.Stream()
            for b in sorted(self.sizes, reverse=True):
                x = self.x[:b]
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):                                          # warm: cuFFT plans, kernel choices
                        self._forward(x)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=cap):
                    out = self._forward(x)
                self.graphs[b] = (g, out)
            torch.cuda.synchronize()

    def run(self, waves):
        """(n, 160000) float32 -> (members, n, 527) float32 probabilities, in AudioSet's order."""
        out = []
        with torch.cuda.device(self.device):
            for i in range(0, len(waves), self.sizes[-1]):
                w = waves[i:i + self.sizes[-1]]
                n = len(w)
                b = next(s for s in self.sizes if s >= n)
                self.host[:n].copy_(torch.from_numpy(np.ascontiguousarray(w, np.float32)))
                self.x[:n].copy_(self.host[:n], non_blocking=True)
                self.x[n:b].zero_()
                if b in self.graphs:
                    g, o = self.graphs[b]
                    g.replay()
                else:
                    o = self._forward(self.x[:b])
                out.append(o[:, :n].cpu().numpy())
        return np.concatenate(out, 1)


class _Job:
    """The windows of one request, filled in as their batches finish, in any order and from any GPU."""

    def __init__(self, n, members):
        self.left, self.out, self.future = n, np.empty((members, n, 527), np.float32), Future()
        self.lock = threading.Lock()

    def put(self, i, p):
        self.out[:, i] = p
        with self.lock:
            self.left -= 1
            done = self.left == 0
        if done and not self.future.done():
            self.future.set_result(self.out)

    def fail(self, e):
        if not self.future.done():
            self.future.set_exception(e)


class Batcher:
    """One queue of windows from every request. A thread per replica takes whatever has queued, up to its largest
    batch; when the queue runs dry it waits at most max_wait_ms for more, so a lone request is not held back for long
    and a busy GPU never waits for a batch to fill."""

    def __init__(self, replicas, max_wait_ms=5.0):
        self.q, self.max_wait, self.replicas = queue.Queue(), max_wait_ms / 1000.0, replicas
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "windows": 0, "batches": 0, "busy_s": 0.0, "errors": 0}
        self.started = time.time()
        self.threads = [threading.Thread(target=self._work, args=(r,), daemon=True, name=f"melati-{r.device}")
                        for r in replicas]
        for t in self.threads:
            t.start()

    def submit(self, windows):
        """(n, 160000) float32, n >= 1 -> a Future of (members, n, 527)."""
        job = _Job(len(windows), len(self.replicas[0].names))
        with self.lock:
            self.stats["requests"] += 1
        for i, w in enumerate(windows):
            self.q.put((w, job, i))
        return job.future

    def close(self):
        """Each GPU's thread finishes what it holds and stops; what is still queued is never run."""
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()

    def _take(self, cap):
        items = [self.q.get()]
        if items[0] is None:                                    # close()
            return None
        deadline = time.monotonic() + self.max_wait
        while len(items) < cap:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    items.append(self.q.get(timeout=left))
                except queue.Empty:
                    break
            if items[-1] is None:                               # close(): run this batch, then stop
                self.q.put(items.pop())
                break
        return items

    def _work(self, rep):
        while True:
            items = self._take(rep.sizes[-1])
            if items is None:
                return
            t0 = time.perf_counter()
            try:
                out = rep.run(np.stack([w for w, _, _ in items]))
            except Exception as e:                  # noqa: BLE001 - a failed batch fails its own requests only
                with self.lock:
                    self.stats["errors"] += len(items)
                for _, job, _ in items:
                    job.fail(e)
                continue
            with self.lock:
                self.stats["windows"] += len(items)
                self.stats["batches"] += 1
                self.stats["busy_s"] += time.perf_counter() - t0
            for k, (_, job, i) in enumerate(items):
                job.put(i, out[:, k])

    def report(self):
        with self.lock:
            s = dict(self.stats)
        up = time.time() - self.started
        s.update(mean_batch=round(s["windows"] / max(s["batches"], 1), 2), queued=self.q.qsize(),
                 gpu_busy=round(s["busy_s"] / max(up * len(self.threads), 1e-9), 3), up_s=round(up, 1))
        return s


class Engine:
    """The ensemble on every GPU in gpus, behind one Batcher.

    path        the repo, downloaded (default: the folder this file is in, when it is the repo's; else
                snapshot_download of REPO at revision)
    gpus        CUDA device indices
    max_batch   the largest batch a GPU runs (windows)
    max_wait_ms how long a GPU waits for a batch to fill when the queue runs dry
    graphs      capture CUDA graphs (False: run eagerly)
    """

    def __init__(self, path=None, gpus=(0,), max_batch=64, max_wait_ms=5.0, graphs=True, revision=None):
        if path is None and revision is None and (HERE / "ensemble.json").exists():
            path = HERE
        if path is None:
            from huggingface_hub import snapshot_download
            path = snapshot_download(REPO, revision=revision)
        self.path = Path(path)
        self.labels = [c["name"] for c in json.loads((self.path / "labels.json").read_text())]
        t0 = time.time()
        self.replicas = [Replica(self.path, f"cuda:{gpus[0]}", max_batch)]        # downloads what is missing
        with ThreadPoolExecutor(max(1, len(gpus) - 1)) as ex:
            self.replicas += list(ex.map(lambda g: Replica(self.path, f"cuda:{g}", max_batch), gpus[1:]))
        if graphs:
            for r in self.replicas:
                r.capture()
        self.load_s = round(time.time() - t0, 1)
        self.names, self.gpus, self.max_batch = self.replicas[0].names, list(gpus), max_batch
        self.batcher = Batcher(self.replicas, max_wait_ms)

    def close(self):
        """Stop the GPUs' threads and free the GPUs (graphs, weights, cache)."""
        import gc
        self.batcher.close()
        self.replicas, self.batcher.replicas = [], []
        gc.collect()
        torch.cuda.empty_cache()

    def submit(self, wave, hop_s=5.0):
        """16 kHz mono float32 samples -> (window starts in seconds, a Future of (members, windows, 527))."""
        starts, W = cut(wave, hop_s)
        return starts, self.batcher.submit(W)

    def tag(self, wave, hop_s=5.0):
        """16 kHz mono float32 samples -> (window starts in seconds, (windows, 527) the members' average, (members,
        windows, 527) each member's)."""
        starts, fut = self.submit(wave, hop_s)
        per = fut.result()
        return starts, per.mean(0), per

    def top(self, p, k=10, threshold=None):
        idx = np.argsort(-p)[:k] if threshold is None else [i for i in np.argsort(-p) if p[i] >= threshold][:k]
        return [{"label": self.labels[i], "index": int(i), "prob": round(float(p[i]), 4)} for i in idx]

    def result(self, starts, per, top_k=10, threshold=None, windows=False, members=False, probs=False):
        """A request's answer from its windows' probabilities: the sounds most probable anywhere in it (the highest over
        its windows), and on request each window's, each member's and all 527."""
        win = per.mean(0)
        clip = win.max(0)
        out = {"windows": len(starts), "tags": self.top(clip, top_k, threshold)}
        if probs:
            out["probs"] = [round(float(v), 4) for v in clip]
        if windows:
            out["per_window"] = [{"start_s": float(s), "tags": self.top(p, top_k, threshold)}
                                 for s, p in zip(starts, win)]
        if members:
            out["members"] = {n: [round(float(v), 4) for v in per[k].max(0)] for k, n in enumerate(self.names)}
        return out


# ------------------------------------------------------------------------------------------------ HTTP
def decode(body, content_type):
    """A request body -> 16 kHz mono float32: an audio file (anything soundfile reads), or JSON
    {"pcm16" or "f32": base64 little-endian samples, "sample_rate": 16000}."""
    if content_type.startswith("application/json"):
        d = json.loads(body)
        if "pcm16" in d:
            x = np.frombuffer(base64.b64decode(d["pcm16"]), "<i2").astype(np.float32) / 32768.0
        else:
            x = np.frombuffer(base64.b64decode(d["f32"]), "<f4").astype(np.float32)
        return resample(torch.from_numpy(x.copy()), int(d.get("sample_rate", SR))).numpy()
    return read(io.BytesIO(body))


def app(engine, name):
    """The HTTP app: POST /tag (the audio as the body), GET /health, /labels and /stats."""
    from fastapi import FastAPI, HTTPException, Request
    api = FastAPI(title="Melati-AudioSet-Ensemble")

    @api.post("/tag")
    async def tag(request: Request, top_k: int = 10, hop_s: float = 5.0, threshold: float | None = None,
                  windows: bool = False, members: bool = False, probs: bool = False):
        body = await request.body()
        try:
            wave = await asyncio.to_thread(decode, body, request.headers.get("content-type", ""))
        except Exception as e:                      # noqa: BLE001 - a body that is not audio is the caller's error
            raise HTTPException(400, f"cannot read the audio: {type(e).__name__}: {e}")
        if not 0.1 <= hop_s <= 10.0:
            raise HTTPException(422, "hop_s is between 0.1 and 10 seconds")
        starts, fut = engine.submit(wave, hop_s)
        per = await asyncio.wrap_future(fut)
        return {"model": name, "duration_s": round(len(wave) / SR, 3), "hop_s": hop_s,
                **engine.result(starts, per, top_k, threshold, windows, members, probs)}

    @api.get("/health")
    async def health():
        return {"ok": True, "model": name, "gpus": engine.gpus, "max_batch": engine.max_batch,
                "members": engine.names}

    @api.get("/labels")
    async def labels():
        return engine.labels

    @api.get("/stats")
    async def stats():
        return engine.batcher.report()
    return api


# ------------------------------------------------------------------------------------------------ offline
def files(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO)
        else:
            yield p


def _read(path):
    try:
        return read(str(path))
    except Exception as e:                          # noqa: BLE001 - one unreadable file is a row, not a crash
        return e


def tag_files(engine, paths, out, hop_s=5.0, top_k=10, probs=False, windows=False, workers=8, inflight=256):
    """Every audio file in paths (directories walked) -> one JSON line each in out, in order: decoded on `workers`
    threads ahead of the GPUs, at most `inflight` files waiting on them."""
    todo = queue.Queue(maxsize=inflight)

    def produce():
        it = iter(files(paths))
        with ThreadPoolExecutor(workers) as ex:
            ahead = collections.deque((p, ex.submit(_read, p)) for p in itertools.islice(it, workers * 4))
            while ahead:
                p, f = ahead.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    ahead.append((nxt, ex.submit(_read, nxt)))
                wave = f.result()
                if isinstance(wave, Exception):
                    todo.put((p, None, None, wave))
                else:
                    starts, fut = engine.submit(wave, hop_s)
                    todo.put((p, len(wave), starts, fut))
        todo.put(None)

    threading.Thread(target=produce, daemon=True).start()
    n, t0 = 0, time.time()
    with open(out, "w") as f:
        while (item := todo.get()) is not None:
            p, length, starts, fut = item
            if length is None:
                row = {"path": str(p), "error": f"{type(fut).__name__}: {fut}"}
            else:
                row = {"path": str(p), "duration_s": round(length / SR, 3),
                       **engine.result(starts, fut.result(), top_k, None, windows, False, probs)}
            f.write(json.dumps(row) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"{n:,} files, {n / (time.time() - t0):.1f}/s", flush=True)
    print(f"{n:,} files in {time.time() - t0:.1f} s -> {out}", flush=True)


def bench(engine, clips=4096, latency_runs=50):
    """Random 10 s clips: the throughput with everything queued at once, and one clip's latency through an idle
    engine (p50, p95), in milliseconds."""
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((min(clips, 512), CLIP)) * 0.1).astype(np.float32)
    lat = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        engine.batcher.submit(W[:1]).result()
        lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    futs = [engine.batcher.submit(W[i % len(W):i % len(W) + 1]) for i in range(clips)]
    for f in futs:
        f.result()
    s = time.perf_counter() - t0
    return {"gpus": engine.gpus, "max_batch": engine.max_batch, "clips": clips, "clips_per_s": round(clips / s, 1),
            "realtime": round(clips * 10 / s), "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 1), "stats": engine.batcher.report()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["serve", "tag", "bench"])
    ap.add_argument("paths", nargs="*", help="tag: audio files or directories")
    ap.add_argument("--gpus", default="0", help="comma-separated CUDA device indices")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--max-wait-ms", type=float, default=5.0)
    ap.add_argument("--no-graphs", action="store_true", help="run eagerly, no CUDA graphs")
    ap.add_argument("--path", default=None, help="the repo, downloaded (default: this file's folder, else the Hub)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="tags.jsonl")
    ap.add_argument("--hop-s", type=float, default=5.0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--probs", action="store_true", help="tag: all 527 probabilities in each row")
    ap.add_argument("--windows", action="store_true", help="tag: each window's tags in each row")
    ap.add_argument("--workers", type=int, default=8, help="tag: threads decoding audio")
    a = ap.parse_args()
    engine = Engine(a.path, [int(g) for g in a.gpus.split(",")], a.max_batch, a.max_wait_ms, not a.no_graphs,
                    a.revision)
    print(f"loaded on GPUs {engine.gpus} in {engine.load_s} s", flush=True)
    if a.command == "serve":
        import uvicorn
        name = f"{REPO}@{a.revision or 'main'}"
        uvicorn.run(app(engine, name), host=a.host, port=a.port, log_level="warning")
    elif a.command == "tag":
        tag_files(engine, a.paths, a.out, a.hop_s, a.top_k, a.probs, a.windows, a.workers)
    else:
        print(json.dumps(bench(engine), indent=1), flush=True)


if __name__ == "__main__":
    main()
