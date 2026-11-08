"""Melati's own AudioSet tagger behind HTTP, with dynamic batching: a probability for each of AudioSet's 527 sounds
in every clip it is sent (Scicom-intl/Melati-AudioSet-EAT-Large, audioset/release.py, or the six-model average
Scicom-intl/Melati-AudioSet-Ensemble, audioset/release_ensemble.py), and, for the inline sound tags, where in a clip
each sound is.

    CUDA_VISIBLE_DEVICES=2 SOUNDS_REVISION=<commit> uvicorn serve.sounds_server:app --port 18700

POST /sounds  {"items": [{"pcm16": b64 at 16 kHz, up to window_s}]}
           -> {"model": "<repo>@<revision>", "window_s": ..., "labels": [...527], "probs": [[527 floats] per item]}
POST /frames  {"items": [{"pcm16": b64 at 16 kHz, any length}], "classes": [AudioSet names]}
           -> {"model": ..., "step_s": 0.16, "classes": [...], "frames": [[[a float per class] per step] per item]}
GET  /health (with window_s, the clip length the tagger reads), /stats

/sounds reads a clip the way the model card's Use snippet reads one. An EAT model: its own Kaldi filterbanks (128
bins every 10 ms), padded with zeros to 1024 frames, so a short clip is not first padded with digital silence, and a
longer clip cut at 10.24 s. The ensemble: its own code from the repo at the pinned revision (melati_ensemble.py, every
member in plain PyTorch), 10 s clips, a shorter one read as it is (SOUNDS_FILL; zeros, as its members were scored,
make it hear Silence in most short segments). Windowing a segment
is the caller's (melati.pipeline.sounds, in the window /health gives). bf16 (autocast for an EAT model, the ensemble's
weights held in bf16 as melati_engine.py serves it), which costs these taggers at most a tenth of an mAP point
(results/audioset/teachers_eval_{bf16,fp32}.json, results/audioset/ensemble_fp32.json).

/frames is the timing audioset/timing_eval.py measured best on AudioSet's strongly labelled test clips (frame mAP
0.4410): every 160 ms, the mean of PretrainedSED's frame taggers (BEATs+ATST-F+M2D, its 40 ms frames pooled by their
maximum) and the frame head trained beside v3 (audioset/frames_train.py, FRAMES_CKPT), over 10.24 s pieces. Only the
classes PretrainedSED has; v4 alone places sounds far worse (0.2970). The frame head is not on the Hub: where
FRAMES_CKPT does not exist the server loads no frame models and /frames answers 404, so the sounds stage writes no
inline tags.

Runs in .venv-sed: EAT's remote code needs transformers 4.46 (5.x breaks it), and that version loads it only from
the Hub, so the model is a repo and a pinned revision, never a local folder. Env: SOUNDS_MODEL, SOUNDS_REVISION,
SOUNDS_FRAMES (1: load the frame models, if FRAMES_CKPT exists), FRAMES_CKPT, MAX_BATCH (64), MAX_WAIT_MS (10),
SOUNDS_BF16 (1), SOUNDS_FILL (the ensemble's short clips: native, or zeros).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from serve.batching import DynamicBatcher, dec_pcm16

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

app = FastAPI(title="melati sounds: AudioSet tagger")
state = {}
SR, FRAMES, STEP_S = 16000, 1024, 0.16
NATIVE_MIN_S = 0.5                          # the ensemble's shortest clip read as it is (SOUNDS_FILL=native)
PIECE = int(10.24 * SR)                     # what one pass of the frame head covers: 64 steps of 160 ms
REPO = Path(__file__).resolve().parents[1]


@app.on_event("startup")
async def startup():
    import torch
    from huggingface_hub import file_exists
    repo = os.environ.get("SOUNDS_MODEL", "Scicom-intl/Melati-AudioSet-EAT-Large")
    rev = os.environ.get("SOUNDS_REVISION") or None
    bf16 = os.environ.get("SOUNDS_BF16", "1") == "1"
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    load = ensemble if file_exists(repo, "ensemble.json", revision=rev) else eat
    tag, labels, window_s, how = load(repo, rev, bf16)
    assert len(labels) == 527, len(labels)
    b = DynamicBatcher(tag, int(os.environ.get("MAX_BATCH", 64)), float(os.environ.get("MAX_WAIT_MS", 10)), "sounds")
    b.start()
    state.update(batcher=b, labels=labels, model=f"{repo}@{rev or 'main'}{how}", window_s=window_s)
    ckpt = os.environ.get("FRAMES_CKPT", f"{MELATI_DATA}/models/audioset/frames1/best.pt")
    if os.environ.get("SOUNDS_FRAMES", "1") == "1":
        if Path(ckpt).exists():
            state["frames_batcher"], state["frames_model"] = frame_models(labels, bf16, ckpt)
        else:                                   # trained by audioset/frames_train.py, which needs AudioSet itself
            print(f"[sounds] no frame models, so no inline sound tags: {ckpt} does not exist", flush=True)


def eat(repo, rev, bf16):
    """An EAT model (Melati-AudioSet-EAT-Large) through its remote code -> (clips -> [527 probabilities each], the
    labels, the clip length it reads in seconds, what the model string adds)."""
    import torch
    import torchaudio
    from transformers import AutoModel
    m = AutoModel.from_pretrained(repo, trust_remote_code=True, revision=rev).eval().cuda()
    labels = [m.config.id2label[i] for i in range(len(m.config.id2label))]      # the config's keys load as ints

    @torch.no_grad()
    def tag(clips):
        fb = []
        for x in clips:
            w = torch.as_tensor(x, dtype=torch.float32, device="cuda")
            w = torch.nn.functional.pad(w, (0, max(0, 400 - len(w))))        # one 25 ms window at least
            f = torchaudio.compliance.kaldi.fbank((w - w.mean())[None], htk_compat=True, sample_frequency=SR,
                                                  use_energy=False, window_type="hanning", num_mel_bins=128,
                                                  dither=0.0, frame_shift=10)
            fb.append(torch.nn.functional.pad(f, (0, 0, 0, max(0, FRAMES - len(f))))[:FRAMES])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = m(((torch.stack(fb) + 4.268) / (4.569 * 2))[:, None])
        logits = out if torch.is_tensor(out) else (out.logits if hasattr(out, "logits") else out[0])
        return list(torch.sigmoid(logits.float()).cpu().numpy())
    return tag, labels, FRAMES / 100, ""


def ensemble(repo, rev, bf16):
    """Melati-AudioSet-Ensemble: the repo's own code at the pinned revision, the six members' average -> (clips ->
    [527 probabilities each], the labels, 10 s). How a clip shorter than 10 s fills the window (SOUNDS_FILL), measured
    on AudioSet's strongly labelled clips cut to a segment's length (audioset/short_eval.py,
    results/audioset/short_eval.json): `native` (the default) gives each member its own features of the clip as it is,
    the EAT members' filterbanks padded to their 1024 frames as v4's are here, CED and BEATs the clip's own length (one
    clip at a time, since their input is not padded); `zeros` pads the clip with digital silence, as the ensemble was
    scored on AudioSet, and then it hears Silence in half the 1 s cuts. Repeating a clip to fill the window scored
    worst of the three (0.25 mAP on 1 s cuts against 0.35)."""
    import torch
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo, revision=rev)
    sys.path.insert(0, path)
    from melati_ensemble import CLIP, SR as ESR, Ensemble, features
    assert ESR == SR, ESR
    fill = os.environ.get("SOUNDS_FILL", "native")
    assert fill in ("native", "zeros"), fill
    m = Ensemble(path, device="cuda")
    if bf16:
        m.bf16_weights()
    kinds = {k.feature for k in m.members.values()}

    @torch.no_grad()
    def native(clips):
        # at least NATIVE_MIN_S, so CED's and BEATs's 16-frame patches fit (shorter: zeros up to it)
        n = int(NATIVE_MIN_S * SR)
        feats = [features(torch.as_tensor(np.pad(x[:CLIP], (0, max(0, n - len(x)))), dtype=torch.float32,
                                          device=m.device)[None], kinds) for x in clips]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=m.bf16):
            per = [k({"eat": torch.cat([f["eat"] for f in feats])}) if k.feature == "eat"
                   else torch.cat([k(f) for f in feats]) for k in m.members.values()]
        return list(torch.stack(per).mean(0).float().cpu().numpy())

    tag = native if fill == "native" else (lambda clips: list(m(clips, batch=len(clips))))
    return tag, m.labels, CLIP / SR, f", {fill} fill"


def frame_models(labels, bf16, ckpt):
    """The two frame models, averaged where both have a class -> (a started DynamicBatcher of clips -> (steps, 527)
    arrays, NaN in the classes PretrainedSED does not have; what it is, in words)."""
    import torch
    from transformers import AutoModel
    sys.path[:0] = [str(REPO / "audioset"), str(REPO / "nonverbal")]
    from audioset_eval import Tagger, class_index
    from frames_train import FrameEAT
    from train import features
    from zoo import REPOS
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    head = FrameEAT(AutoModel.from_pretrained(REPOS[ck["init"]], trust_remote_code=True))
    head.load_state_dict(ck["state"])
    head = head.cuda().eval()
    psed = Tagger("BEATs+ATST-F+M2D", batch=32)
    ours = {n: i for i, n in enumerate(labels)}
    pairs = [(theirs, ours[n]) for n, theirs in class_index().items() if n in ours]

    @torch.no_grad()
    def frames(clips):
        pieces, owner = [], []
        for k, x in enumerate(clips):
            for i in range(max(1, math.ceil(len(x) / PIECE))):
                c = np.asarray(x[i * PIECE:(i + 1) * PIECE], np.float32)
                pieces.append(np.pad(c, (0, PIECE - len(c)))); owner.append(k)
        H = []
        for i in range(0, len(pieces), 32):
            w = torch.from_numpy(np.stack(pieces[i:i + 32])).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                H.append(torch.sigmoid(head(features(w))[1].float()).cpu().numpy())
        H = np.concatenate(H)                                               # (pieces, 64, 527)
        out = []
        for k, (x, p) in enumerate(zip(clips, psed([np.asarray(c, np.float32) for c in clips]))):
            steps = max(1, math.ceil(len(x) / SR / STEP_S))
            h = np.concatenate([H[j] for j, o in enumerate(owner) if o == k])[:steps]
            p = np.pad(p, ((0, 0), (0, max(0, 4 * steps - p.shape[1]))))[:, :4 * steps]   # 40 ms frames
            p = p.reshape(p.shape[0], steps, 4).max(2)                        # -> 160 ms steps
            mean = np.full((steps, len(labels)), np.nan, np.float32)
            for theirs, j in pairs:
                mean[:, j] = (p[theirs] + h[:, j]) / 2
            out.append(mean)
        return out

    b = DynamicBatcher(frames, 16, float(os.environ.get("MAX_WAIT_MS", 10)), "frames")
    b.start()
    return b, f"mean of PretrainedSED BEATs+ATST-F+M2D and the frame head of {ckpt}"


@app.post("/sounds")
async def sounds(body: dict):
    probs = await state["batcher"].submit_many([dec_pcm16(it["pcm16"]) for it in body["items"]])
    return {"model": state["model"], "window_s": state["window_s"], "labels": state["labels"],
            "probs": [[round(float(v), 4) for v in p] for p in probs]}


@app.post("/frames")
async def frames(body: dict):
    if "frames_batcher" not in state:
        raise HTTPException(404, "the frame models are not loaded here (SOUNDS_FRAMES=0, or no FRAMES_CKPT)")
    idx = [state["labels"].index(c) for c in body["classes"]]
    out = await state["frames_batcher"].submit_many([dec_pcm16(it["pcm16"]) for it in body["items"]])
    if any(np.isnan(a[:, idx]).any() for a in out):
        raise HTTPException(422, "a class PretrainedSED does not have: no frame timing for it")
    return {"model": state["frames_model"], "step_s": STEP_S, "classes": body["classes"],
            "frames": [[[round(float(v), 3) for v in row] for row in a[:, idx]] for a in out]}


@app.get("/health")
async def health():
    return {"ok": "batcher" in state, "model": state.get("model"), "window_s": state.get("window_s"),
            "frames": "frames_batcher" in state}


@app.get("/stats")
async def stats():
    return {k: v.report() for k, v in state.items() if isinstance(v, DynamicBatcher)}
