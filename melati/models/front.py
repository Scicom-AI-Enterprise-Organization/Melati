"""The two GPU models of Stage A, as batch functions for serve/front_server.py.

  PANNs Cnn14_DecisionLevelMax  frame-level AudioSet events (melati.sed.GROUPS), 100 frames/s at
                                32 kHz; one item is a chunk of up to 60 s
  DBS-TIGER (DnR v2)            music / effects / speech at 44.1 kHz; one item is one 6 s window

A server batches items from many requests, so a batch of separation windows can hold windows of
several recordings: the GPU sees full batches no matter how short each region is.
"""
from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np
import torch

from melati import pins
from melati.sed import GROUPS

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

SED_SR, SED_FPS, SED_CHUNK_S = 32000, 100, 60.0
SEP_SR, SEP_CHUNK_S, SEP_HOP_S = 44100, 6.0, 3.0


class PANNsSED:
    def __init__(self, model_dir=f"{MELATI_DATA}/models/panns", device="cuda"):
        from panns_inference.models import Cnn14_DecisionLevelMax
        model_dir = Path(model_dir)
        names = [r["display_name"] for r in csv.DictReader((model_dir / "class_labels_indices.csv").open())]
        self.idx = {g: [names.index(m) for m in ms if m in names] for g, ms in GROUPS.items()}
        self.idx = {g: ix for g, ix in self.idx.items() if ix}
        self.model = Cnn14_DecisionLevelMax(sample_rate=SED_SR, window_size=1024, hop_size=320, mel_bins=64,
                                            fmin=50, fmax=14000, classes_num=527)
        ck = torch.load(model_dir / "Cnn14_DecisionLevelMax.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, chunks):
        """[chunk float32 at 32 kHz, <= 60 s] -> [{group: float16 [frames]}]."""
        n = int(SED_CHUNK_S * SED_SR)
        lens = [len(c) for c in chunks]
        x = np.zeros((len(chunks), max(max(lens), SED_SR // 2)), np.float32)
        for i, c in enumerate(chunks):
            x[i, :len(c)] = c[:n]
        fw = self.model(torch.from_numpy(x).to(self.device))["framewise_output"].float().cpu().numpy()
        out = []
        for i, L in enumerate(lens):
            f = fw[i, : int(np.ceil(L / 320))]
            out.append({g: f[:, ix].max(1).astype(np.float16) for g, ix in self.idx.items()})
        return out


class DBSSeparator:
    def __init__(self, repo="JusperLee/DBS-TIGER-DnR", device="cuda"):
        from dbs.models import DBSTIGER
        self.model = DBSTIGER.from_pretrained(repo, revision=pins.rev(repo)).to(device).eval()
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def __call__(self, windows):
        """[window float32 at 44.1 kHz, exactly 6 s] -> [float32 [3, 6 s]] (music, sfx, speech)."""
        w = torch.from_numpy(np.stack(windows)).to(self.device).unsqueeze(1)
        est = self.model(w).float().cpu().numpy()
        return list(est)


def windows_of(x: np.ndarray):
    """Cut a region into the separator's 6 s windows every 3 s (zero-padded at the end)."""
    chunk, hop = int(SEP_SR * SEP_CHUNK_S), int(SEP_SR * SEP_HOP_S)
    n = len(x)
    starts = [0] if n <= chunk else list(range(0, n - chunk + hop, hop))
    if n > chunk and starts[-1] + chunk < n:
        starts.append(starts[-1] + hop)
    wins = []
    for s in starts:
        w = np.zeros(chunk, np.float32)
        piece = x[s:s + chunk]
        w[:len(piece)] = piece
        wins.append(w)
    return starts, wins


def overlap_add(n: int, starts, estimates):
    """Uniform-weight overlap-add of window estimates: dbs.engine.chunked's arithmetic."""
    out = np.zeros((3, n), np.float32)
    weight = np.zeros(n, np.float32)
    for s, est in zip(starts, estimates):
        valid = min(est.shape[-1], n - s)
        out[:, s:s + valid] += est[:, :valid]
        weight[s:s + valid] += 1.0
    return out / np.maximum(weight, 1.0)
