"""Dynamic batching for the model servers, and the audio codec their JSON speaks.

Emilia-Pipe scales by running the whole pipeline once per GPU worker: every worker holds a copy of
every model and walks each file through every stage in turn, so each model sees one file's
segments at a time (the Scicom fork measured a mean Whisper batch of 2.29 segments). Melati puts
each GPU model behind its own server instead. A server turns many concurrent requests into full
GPU batches, and a model is scaled by giving it more replicas (or, for a vLLM model, a larger
tensor- or data-parallel size), independently of every other model and of the CPU work around it.

    batcher = DynamicBatcher(fn, max_batch=32, max_wait_ms=10)   # fn(list[item]) -> list[result]
    result = await batcher.submit(item)

One GPU thread per process runs `fn`; while it runs, new requests queue, and the next batch is
everything that queued (up to max_batch). Under load the GPU never waits for a batch to fill;
when idle a lone request waits at most max_wait_ms.
"""
from __future__ import annotations

import asyncio
import base64
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class DynamicBatcher:
    def __init__(self, fn, max_batch: int = 32, max_wait_ms: float = 10.0, name: str = "model"):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000.0
        self.name = name
        self.queue: asyncio.Queue | None = None
        self.gpu = ThreadPoolExecutor(1, thread_name_prefix=f"{name}-gpu")
        self.stats = {"requests": 0, "items": 0, "batches": 0, "gpu_s": 0.0, "errors": 0}
        self._task = None

    def start(self):
        self.queue = asyncio.Queue()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def submit(self, item):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def submit_many(self, items, raise_errors: bool = True):
        """Results in order. An item that failed is an Exception in the list, or raises if asked."""
        self.stats["requests"] += 1
        res = await asyncio.gather(*[self.submit(it) for it in items], return_exceptions=True)
        if raise_errors:
            for r in res:
                if isinstance(r, Exception):
                    raise r
        return res

    async def _take(self):
        loop = asyncio.get_running_loop()
        batch = [await self.queue.get()]
        deadline = loop.time() + self.max_wait
        while len(batch) < self.max_batch:
            try:
                batch.append(self.queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            left = deadline - loop.time()
            if left <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), left))
            except asyncio.TimeoutError:
                break
        return batch

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._take()
            items, futs = zip(*batch)
            t0 = time.perf_counter()
            try:
                res = await loop.run_in_executor(self.gpu, self.fn, list(items))
                if len(res) != len(batch):          # an item with no result would leave its request waiting for ever
                    raise RuntimeError(f"{self.name}: {len(res)} results for a batch of {len(batch)}")
            except Exception as ex:  # noqa: BLE001 - one bad batch fails its own requests only
                self.stats["errors"] += len(batch)
                for f in futs:
                    if not f.done():
                        f.set_exception(ex)
                continue
            self.stats["gpu_s"] += time.perf_counter() - t0
            self.stats["batches"] += 1
            self.stats["items"] += len(batch)
            for f, r in zip(futs, res):
                if not f.done():
                    if isinstance(r, Exception):
                        f.set_exception(r)
                    else:
                        f.set_result(r)

    def report(self):
        s = dict(self.stats)
        s["mean_batch"] = round(s["items"] / max(s["batches"], 1), 2)
        s["queue"] = self.queue.qsize() if self.queue else 0
        return s


# ---------------------------------------------------------------------------- audio over JSON
def enc_pcm16(x: np.ndarray) -> str:
    """float32 [-1, 1] -> base64 of little-endian int16."""
    return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def dec_pcm16(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<i2").astype(np.float32) / 32767.0


def enc_f16(x: np.ndarray) -> str:
    return base64.b64encode(np.asarray(x, dtype="<f2").tobytes()).decode()


def dec_f16(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f2").astype(np.float32)
