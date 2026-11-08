"""Non-verbal sounds frame by frame: laughter and coughs over speech too, where the gap finder (melati.nv_gaps) cannot
reach. Trained and scored by nonverbal/frames.py; served by the tag server as `nvf_<encoder>`.

The frozen openai/whisper-large-v3 encoder, eight of its layers averaged to 10 frames a second, a learned mix of
them, LayerNorm, a projection and four residual dilated convolutions over time (about 6 s of context): every 0.1 s,
an independent probability of laughter, a cough, another vocal sound, and speech. A segment longer than 30 s is read
in 30 s windows. An event is a run of frames over a threshold, joined across up to two quiet frames, at least two
frames long.
"""
from __future__ import annotations

import os

import numpy as np

from melati import pins

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

SR = 16000
WIN_S = 30.0
FPS = 10                                  # 50 encoder frames a second, averaged in fives
T = int(WIN_S * FPS)
LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)   # hidden_states indices; evenly spaced, nothing tuned on a test set
CLASSES = ("laugh", "cough", "other", "speech")
NV_FRAMES_DIR = os.environ.get("MELATI_NV_FRAMES", f"{MELATI_DATA}/models/nv_frames")


def encoder(device="cuda", numpy=True):
    """clips (16 kHz, at most 30 s) -> (B, T, len(LAYERS), 1280) float16 and the frames each clip covers. The mean
    over five encoder frames is taken in float16, as the training features were."""
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
    name = "openai/whisper-large-v3"
    fe = WhisperFeatureExtractor.from_pretrained(name, revision=pins.rev(name))
    enc = WhisperModel.from_pretrained(name, revision=pins.rev(name), dtype=torch.float16).encoder.to(device).eval()

    def run(batch):
        with torch.inference_mode():
            f = fe(batch, sampling_rate=SR, return_tensors="pt").input_features.to(device).half()
            hs = enc(f, output_hidden_states=True).hidden_states
            h = torch.stack([hs[k] for k in LAYERS], 2)                    # B, 1500, L, D
            h = h.view(h.shape[0], T, 1500 // T, *h.shape[2:]).mean(2)     # B, T, L, D
            n = [min(T, int(np.ceil(len(x) / SR * FPS))) for x in batch]
            return (h.cpu().numpy().astype(np.float16) if numpy else h), n
    return run


def frame_head(layers=len(LAYERS), dim=1280, hidden=256, n=len(CLASSES), k=5, dilations=(1, 2, 4, 8), drop=0.1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FrameHead(nn.Module):
        """A softmax mix of the layers, LayerNorm, a projection, then residual dilated convolutions over time."""

        def __init__(self):
            super().__init__()
            self.mix = nn.Parameter(torch.zeros(layers))
            self.norm = nn.LayerNorm(dim)
            self.inp = nn.Linear(dim, hidden)
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in dilations)
            self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, k, padding=d * (k - 1) // 2, dilation=d)
                                       for d in dilations)
            self.drop = nn.Dropout(drop)
            self.out = nn.Linear(hidden, n)

        def forward(self, x):                                   # B, T, L, D -> B, T, n logits
            h = (self.mix.softmax(0)[None, None, :, None] * x).sum(2)
            h = self.drop(F.gelu(self.inp(self.norm(h))))
            for nm, conv in zip(self.norms, self.convs):
                h = h + self.drop(F.gelu(conv(nm(h).transpose(1, 2)).transpose(1, 2)))
            return self.out(h)
    return FrameHead()


def events(p, t, min_frames=2, join=2):
    """Runs of frames with p >= t, joined across up to `join` quiet frames, at least min_frames long -> [(a, b, peak)]
    in frames."""
    on = np.flatnonzero(np.asarray(p) >= t)
    out = []
    for f in on:
        if out and f - out[-1][1] <= join:
            out[-1][1] = f + 1
        else:
            out.append([f, f + 1])
    return [(a, b, float(np.max(p[a:b]))) for a, b in out if b - a >= min_frames]


def events_hysteresis(p, high, low, min_frames=2, join=2):
    """Runs of frames with p >= low, joined across up to `join` frames under it, that reach `high` somewhere and are
    at least min_frames long -> [(a, b, peak)] in frames. Whether there is an event is decided at `high`, how far it
    reaches at `low`: a sound that hovers about one threshold (music under speech) is one event from where it starts,
    not several from wherever it crosses."""
    p = np.asarray(p)
    out = []
    for f in np.flatnonzero(p >= low):
        if out and f - out[-1][1] <= join:
            out[-1][1] = f + 1
        else:
            out.append([f, f + 1])
    return [(a, b, float(np.max(p[a:b]))) for a, b in out if b - a >= min_frames and np.max(p[a:b]) >= high]


class FrameNV:
    """nvf_<encoder>: clips (16 kHz, any length) -> ([per clip (frames, len(CLASSES)) probabilities], None), the
    call melati.emotion's models share."""

    def __init__(self, encoder_name="whisper", device="cuda", models_dir=NV_FRAMES_DIR, sub_batch=16):
        import torch
        ck = torch.load(os.path.join(models_dir, f"{encoder_name}.pt"), map_location="cpu")
        assert tuple(ck["classes"]) == CLASSES and tuple(ck["layers"]) == LAYERS, (ck["classes"], ck["layers"])
        self.head = frame_head().to(device).eval()
        self.head.load_state_dict(ck["state"])
        self.run = encoder(device, numpy=False)
        self.name, self.classes, self.device, self.sub_batch = f"nvf_{encoder_name}", CLASSES, device, sub_batch

    def __call__(self, clips):
        import torch
        wins, owner = [], []
        for i, x in enumerate(clips):
            x = np.asarray(x, np.float32)
            for k in range(max(1, int(np.ceil(len(x) / (WIN_S * SR))))):
                w = x[int(k * WIN_S * SR):int((k + 1) * WIN_S * SR)]
                wins.append(w if len(w) >= SR // 10 else np.pad(w, (0, SR // 10 - len(w))))
                owner.append(i)
        out = [[] for _ in clips]
        for s in range(0, len(wins), self.sub_batch):
            h, n = self.run(wins[s:s + self.sub_batch])
            with torch.inference_mode():
                p = torch.sigmoid(self.head(h.float())).cpu().numpy()
            for q, (k, i) in enumerate(zip(n, owner[s:s + self.sub_batch])):
                out[i].append(p[q, :k])
        return [np.concatenate(o) if o else np.zeros((0, len(CLASSES)), np.float32) for o in out], None
