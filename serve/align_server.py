"""Word start and end times behind HTTP, with dynamic batching.

    CUDA_VISIBLE_DEVICES=4 uvicorn serve.align_server:app --host 127.0.0.1 --port 8400

POST /align  {"items": [{"pcm16": b64 at 16 kHz (<= 30 s), "text": str, "language": "ms" | null}]}
          -> {"results": [{"words": [{"w", "s", "e", "c"}], "score"} | {"error": str}]}
GET  /health, /stats

Each item is one encoder window plus one teacher-forced decoder pass, whatever its length, so the
batcher's job is simply to keep MAX_BATCH items in every GPU call. Env: MAX_BATCH (32),
MAX_WAIT_MS (10), ALIGN_REPO.
"""
from __future__ import annotations

import os

from fastapi import FastAPI

from serve.batching import DynamicBatcher, dec_pcm16

app = FastAPI(title="melati align: Where-Words-Begin-and-End")
state = {}


@app.on_event("startup")
async def startup():
    from melati.models.aligner import REPO, Aligner
    al = Aligner(os.environ.get("ALIGN_REPO", REPO))
    state["align"] = DynamicBatcher(al, int(os.environ.get("MAX_BATCH", 32)),
                                    float(os.environ.get("MAX_WAIT_MS", 10)), "align")
    state["align"].start()


@app.post("/align")
async def align(body: dict):
    items = [(dec_pcm16(it["pcm16"]), it["text"], it.get("language")) for it in body["items"]]
    res = []
    for r in await state["align"].submit_many(items, raise_errors=False):
        res.append({"error": str(r)} if isinstance(r, Exception) or r is None else r)
    return {"results": res}


@app.get("/health")
async def health():
    return {"ok": "align" in state}


@app.get("/stats")
async def stats():
    return state["align"].report()
