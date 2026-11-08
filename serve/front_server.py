"""Stage A's GPU models behind HTTP, with dynamic batching: sound events and separation.

    CUDA_VISIBLE_DEVICES=3 uvicorn serve.front_server:app --host 127.0.0.1 --port 8300

POST /sed       {"pcm16": b64, "sr": 32000}  -> {"fps": 100, "tracks": {group: b64 float16}}
POST /separate  {"pcm16": b64, "sr": 44100}  -> {"sr": 44100, "music": b64 f16, "sfx": ..., "speech": ...}
GET  /health, /stats

A request is cut into items (60 s chunks for /sed, 6 s windows every 3 s for /separate) and each
item is queued on its model's batcher, so one GPU batch holds pieces of many requests. The
response is reassembled from the items (overlap-added for separation).

Env: SED_MAX_BATCH (16), SEP_MAX_BATCH (16), MAX_WAIT_MS (10), PANNS_DIR, DBS_REPO.
"""
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI

from serve.batching import DynamicBatcher, dec_pcm16, enc_f16

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

app = FastAPI(title="melati front: sound events + separation")
state = {}


@app.on_event("startup")
async def startup():
    from melati.models.front import DBSSeparator, PANNsSED
    sed = PANNsSED(os.environ.get("PANNS_DIR", f"{MELATI_DATA}/models/panns"))
    sep = DBSSeparator(os.environ.get("DBS_REPO", "JusperLee/DBS-TIGER-DnR"))
    wait = float(os.environ.get("MAX_WAIT_MS", 10))
    state["sed"] = DynamicBatcher(sed, int(os.environ.get("SED_MAX_BATCH", 16)), wait, "sed")
    state["sep"] = DynamicBatcher(sep, int(os.environ.get("SEP_MAX_BATCH", 16)), wait, "sep")
    state["sed"].start()
    state["sep"].start()


@app.post("/sed")
async def sed(body: dict):
    from melati.models.front import SED_CHUNK_S, SED_FPS, SED_SR
    x = dec_pcm16(body["pcm16"])
    assert int(body.get("sr", SED_SR)) == SED_SR, "send 32 kHz audio"
    n = int(SED_CHUNK_S * SED_SR)
    chunks = [x[a:a + n] for a in range(0, max(len(x), 1), n)]
    parts = await state["sed"].submit_many(chunks)
    tracks = {g: np.concatenate([p[g] for p in parts]) for g in parts[0]}
    return {"fps": SED_FPS, "tracks": {g: enc_f16(v) for g, v in tracks.items()}}


@app.post("/separate")
async def separate(body: dict):
    from melati.models.front import SEP_SR, overlap_add, windows_of
    x = dec_pcm16(body["pcm16"])
    assert int(body.get("sr", SEP_SR)) == SEP_SR, "send 44.1 kHz audio"
    starts, wins = windows_of(x)
    est = await state["sep"].submit_many(wins)
    stems = overlap_add(len(x), starts, est)
    return {"sr": SEP_SR, "music": enc_f16(stems[0]), "sfx": enc_f16(stems[1]), "speech": enc_f16(stems[2])}


@app.get("/health")
async def health():
    return {"ok": "sep" in state}


@app.get("/stats")
async def stats():
    return {k: b.report() for k, b in state.items()}
