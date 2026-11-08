"""Emilia-Pipe's separator, UVR-MDX-Net-Inst_HQ_3, behind HTTP, for the comparisons only.

    CUDA_VISIBLE_DEVICES=5 uvicorn serve.uvr_server:app --port 17700     (venv: deploy/box_uvr_setup.sh)

POST /uvr  {"pcm16": b64 mono at 44.1 kHz} -> {"sr": 44100, "vocals": b64 float16}

It runs upstream's own code (baselines/emilia_upstream/models/separate_fast.py) with upstream's
config (denoise on, 15 s chunks), and returns the stem Emilia keeps: vocals, channel 0. Requests
run one at a time on the GPU (the predictor already chunks its input).
"""
from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI

from serve.batching import dec_pcm16, enc_f16

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

app = FastAPI(title="melati uvr: Emilia-Pipe's separator")
state = {}
UPSTREAM = Path(__file__).resolve().parents[1] / "baselines" / "emilia_upstream"


@app.on_event("startup")
async def startup():
    sys.path.insert(0, str(UPSTREAM))
    from models.separate_fast import Predictor
    state["uvr"] = Predictor(args={"model_path": os.environ.get("UVR_ONNX", f"{MELATI_DATA}/models/uvr/UVR-MDX-NET-Inst_HQ_3.onnx"),
                                   "denoise": True, "margin": 44100, "chunks": 15, "n_fft": 6144, "dim_t": 8,
                                   "dim_f": 3072}, device="cuda")
    state["gpu"] = ThreadPoolExecutor(1)


@app.post("/uvr")
async def uvr(body: dict):
    x = dec_pcm16(body["pcm16"])
    vocals, _ = await asyncio.get_running_loop().run_in_executor(state["gpu"], state["uvr"].predict, x)
    return {"sr": 44100, "vocals": enc_f16(vocals[:, 0])}


@app.get("/health")
async def health():
    return {"ok": "uvr" in state}
