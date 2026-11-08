"""Emotion classifiers and DNSMOS behind HTTP, with dynamic batching.

    CUDA_VISIBLE_DEVICES=5 EMOTION_MODELS=meralion,e2v uvicorn serve.tag_server:app --port 8600

POST /emotion  {"model": "meralion" | "e2v", "items": [{"pcm16": b64 at 16 kHz}]}
            -> {"labels": [...7], "probs": [[7 floats]]}
POST /frames   {"model": "nvf_whisper", "items": [{"pcm16": b64 at 16 kHz}]}
            -> {"classes": [...], "fps": 10, "frames": [[[a float per class] per frame] per item]}
POST /dnsmos   {"items": [{"pcm16": b64 at 16 kHz}]} -> {"scores": [{"OVRL", "SIG", "BAK"}]}
GET  /health, /stats

DNSMOS P.835 is Emilia's quality filter; it is computed here (ONNX, on CPU threads) so the paper
can compare it with SOTA-MOS on the same segments. Env: EMOTION_MODELS (meralion,e2v),
MAX_BATCH (32), MAX_WAIT_MS (10), DNSMOS_ONNX, DNSMOS_THREADS (16), TORCH_THREADS (4).
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException

from serve.batching import DynamicBatcher, dec_pcm16

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

app = FastAPI(title="melati tag: emotion + DNSMOS")
state = {}


@app.on_event("startup")
async def startup():
    import torch
    from melati import emotion as EMO
    from melati.third_party.dnsmos import ComputeScore
    # CPU work here (feature extraction, DNSMOS) runs in many threads at once; torch's default of a thread per core
    # in each of them oversubscribes a box that six of these servers share
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    state["labels"] = list(EMO.LABELS)
    for name in [m for m in os.environ.get("EMOTION_MODELS", "meralion,e2v").split(",") if m]:
        try:
            model = EMO.load(name, "cuda")
        except Exception as e:  # noqa: BLE001 - a missing optional model must not take the server down
            print(f"[tag] {name} not loaded: {type(e).__name__}: {e}", flush=True)
            continue
        b = DynamicBatcher(lambda clips, m=model: list(m(clips)[0]), int(os.environ.get("MAX_BATCH", 32)),
                           float(os.environ.get("MAX_WAIT_MS", 10)), name)
        b.start()
        if name.startswith("nvf_"):                 # frame-level: a sequence per clip, served by /frames
            state[f"frames/{name}"] = b
            state[f"frames_meta/{name}"] = {"classes": list(model.classes), "fps": 10}
        else:
            state[f"emotion/{name}"] = b
    state["dnsmos"] = ComputeScore(os.environ.get("DNSMOS_ONNX", f"{MELATI_DATA}/models/dnsmos/sig_bak_ovr.onnx"),
                                   "cpu")
    state["dnsmos_pool"] = ThreadPoolExecutor(int(os.environ.get("DNSMOS_THREADS", 16)))


@app.post("/emotion")
async def emotion(body: dict):
    b = state.get(f"emotion/{body.get('model', 'meralion')}")
    if b is None:
        raise HTTPException(404, f"emotion model {body.get('model')} is not loaded here")
    probs = await b.submit_many([dec_pcm16(it["pcm16"]) for it in body["items"]])
    return {"labels": state["labels"], "probs": [[round(float(v), 4) for v in p] for p in probs]}


@app.post("/frames")
async def frames(body: dict):
    name = body.get("model", "nvf_whisper")
    b = state.get(f"frames/{name}")
    if b is None:
        raise HTTPException(404, f"frame model {name} is not loaded here")
    out = await b.submit_many([dec_pcm16(it["pcm16"]) for it in body["items"]])
    return {**state[f"frames_meta/{name}"],
            "frames": [[[round(float(v), 3) for v in row] for row in p] for p in out]}


@app.post("/dnsmos")
async def dnsmos(body: dict):
    loop = asyncio.get_running_loop()

    def one(x):
        r = state["dnsmos"](x, 16000, False)
        return {k: round(float(r[k]), 3) for k in ("OVRL", "SIG", "BAK")}

    scores = await asyncio.gather(*[loop.run_in_executor(state["dnsmos_pool"], one, dec_pcm16(it["pcm16"]))
                                    for it in body["items"]])
    return {"scores": scores}


@app.get("/health")
async def health():
    return {"ok": "dnsmos" in state, "emotion": [k.split("/")[1] for k in state if k.startswith("emotion/")],
            "frames": [k.split("/")[1] for k in state if k.startswith("frames/")]}


@app.get("/stats")
async def stats():
    return {k: v.report() for k, v in state.items() if isinstance(v, DynamicBatcher)}
