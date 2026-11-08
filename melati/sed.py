"""Frame-level sound events over a whole recording: what is going on, every 10 ms.

One pass of PANNs Cnn14_DecisionLevelMax (AudioSet, 527 classes, ~100 frames/s at 32 kHz) over the
original mixture answers two questions for the rest of the pipeline:

  background   is there music, a crowd, a television, traffic under the speech here? Separation
               runs only where there is (melati.pipeline.front): DBS-TIGER manages 14x realtime on one
               GPU, and a podcast is mostly a voice in a quiet room.
  vocalisation is someone laughing, coughing, sighing, gasping here? These frames are the
               candidates the non-verbal stage verifies and places (melati.pipeline.tag).

Only the class groups below are kept, as float16 tracks at 100 frames/s (`sed.npz`), so a
threshold can be changed later without running the model again. AudioSet laughter posteriors are
known to be suppressed on laughter that overlaps speech (median clip-max 0.007 on Expresso's
`laughing` style, measured by the earlier Malaysian-Emilia non-verbal work), which is why these
are candidates and precision comes from verification, not from this threshold.
"""
from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

SR = 32000
FPS = 100
MODEL_DIR = Path(f"{MELATI_DATA}/models/panns")

GROUPS = {
    # background: what separation is for
    "music": ["Music", "Musical instrument", "Plucked string instrument", "Guitar", "Piano", "Drum",
              "Drum kit", "Electronic music", "Pop music", "Hip hop music", "Rock music", "Background music",
              "Theme music", "Jingle (music)", "Soundtrack music", "Song", "Synthesizer", "Bass guitar"],
    "singing": ["Singing", "Choir", "Male singing", "Female singing", "Child singing", "Rapping", "Humming"],
    "crowd": ["Crowd", "Chatter", "Hubbub, speech noise, speech babble", "Applause", "Cheering", "Clapping",
              "Children shouting"],
    "media": ["Television", "Radio", "Sound effect", "Video game music"],
    "noise": ["Vehicle", "Car", "Traffic noise, roadway noise", "Wind noise (microphone)", "Rain", "Engine",
              "Air conditioning", "Mechanical fan", "White noise", "Pink noise", "Static", "Hum", "Buzz"],
    # vocalisations: the non-verbal candidates
    "laugh": ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"],
    "cough": ["Cough"],
    "throat_clear": ["Throat clearing"],
    "sneeze": ["Sneeze"],
    "sniff": ["Sniff"],
    "sigh": ["Sigh"],
    "gasp": ["Gasp"],
    "breath": ["Breathing", "Wheeze", "Pant"],
    "cry": ["Crying, sobbing", "Whimper"],
    "yawn": ["Yawn"],
    "scream": ["Screaming", "Yell"],
    "groan": ["Groan", "Grunt"],
    "hum": ["Humming"],
    "whisper": ["Whispering"],
    "shout": ["Shout", "Yell"],
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation",
               "Narration, monologue"],
}
BACKGROUND = ("music", "singing", "crowd", "media", "noise")


class PANNs:
    def __init__(self, device="cuda", model_dir: Path = MODEL_DIR):
        import torch
        from panns_inference.models import Cnn14_DecisionLevelMax
        names = [r["display_name"] for r in csv.DictReader((model_dir / "class_labels_indices.csv").open())]
        self.idx = {}
        for g, members in GROUPS.items():
            ix = [names.index(m) for m in members if m in names]
            if ix:
                self.idx[g] = ix
        self.model = Cnn14_DecisionLevelMax(sample_rate=SR, window_size=1024, hop_size=320, mel_bins=64,
                                            fmin=50, fmax=14000, classes_num=527)
        ck = torch.load(model_dir / "Cnn14_DecisionLevelMax.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device, self.torch = device, torch

    def __call__(self, x32k: np.ndarray, chunk_s: float = 60.0) -> dict:
        """{group: float16 [frames]} at 100 frames/s (max over the group's classes)."""
        torch = self.torch
        n = int(chunk_s * SR)
        parts = {g: [] for g in self.idx}
        with torch.inference_mode():
            for a in range(0, max(len(x32k), 1), n):
                piece = x32k[a:a + n]
                if len(piece) < SR // 2:
                    piece = np.pad(piece, (0, SR // 2 - len(piece)))
                fw = self.model(torch.from_numpy(piece[None]).float().to(self.device))["framewise_output"][0]
                fw = fw[: int(np.ceil(len(x32k[a:a + n]) / 320))].float().cpu().numpy()
                for g, ix in self.idx.items():
                    parts[g].append(fw[:, ix].max(1))
        return {g: np.concatenate(v).astype(np.float16) for g, v in parts.items()}


def regions(track: np.ndarray, thr: float, dilate_s: float = 1.0, min_s: float = 0.5, fps: int = FPS):
    """Spans [(start, end)] in seconds where `track` >= thr, dilated and merged."""
    on = track.astype(np.float32) >= thr
    if not on.any():
        return []
    d = int(dilate_s * fps)
    idx = np.flatnonzero(on)
    spans, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > 2 * d + 1:
            spans.append((s, prev)); s = i
        prev = i
    spans.append((s, prev))
    out = [(max(0, a - d) / fps, (b + 1 + d) / fps) for a, b in spans]
    return [(a, b) for a, b in out if b - a >= min_s]
