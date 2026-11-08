"""Melati's non-verbal classifier: what a sound in a gap between two aligned words is, where laughter and coughs
are over speech, and where to tag them.

The model (modeling_melati_head.py, one checkpoint) is the encoder of openai/whisper-large-v3 with the two heads
Melati's pipeline trained on it (https://github.com/Scicom-AI-Enterprise-Organization/Melati): one classifies a sound
cut out of a word gap, the other reads a whole segment and gives a probability of laughter and of a cough every
0.1 s. This file adds what the pipeline does around them: find the candidates in the word gaps, cut them out the way
the head was trained on, and decide which become tags.

    from melati_nonverbal import NonVerbal
    nv = NonVerbal.from_pretrained("Scicom-intl/Melati-NonVerbal-Whisper-Large-V3")
    probs = nv.classify([clip])                 # (N, 11) over nv.labels; a clip is 16 kHz mono float32
    frames = nv.frames(audio)                   # (frames, 4) over nv.frame_classes, 10 a second
    events = nv.tag(audio, words)               # words: [{"s": start_s, "e": end_s}, ...] from a forced aligner

`tag` finds the candidates (sound in each gap between words, before the first and after the last, above the clip's
noise floor + 2 dB and its speech level - 40 dB for at least 80 ms), cuts each out with 0.3 s of silence on either
side, and keeps the ones whose most probable non-verbal class clears its threshold; then it adds the frame
detector's laughs and coughs (runs of frames over that class's threshold, at least 0.2 s) wherever the gap classifier
found none of the kind within 0.3 s.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 0.01
# what a head label means as a tag class (melati.nv_gaps.TO_CLASS); speech, filler, noise and other are never tags
TO_CLASS = {"breath": "breath", "laugh": "laugh", "cough": "cough", "sigh": "sigh", "sniff": "sniff",
            "throat clearing": "throat_clear", "sneeze": "sneeze"}


def candidates(x, words, rel_db=-40.0, floor_db=2.0, min_active=0.08, dilate=0.03, max_len=2.5):
    """Sound in each word gap of `x` (16 kHz): frames above the noise floor + floor_db AND above the speech level +
    rel_db. `words` are [{"s", "e"}] in seconds. -> [{slot, s, e, level_db}]; slot k = before word k."""
    n = int(FRAME * SR)
    m = len(x) // n
    if m < 10 or not words:
        return []
    db = 10 * np.log10(np.mean(x[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    inside = np.zeros(m, bool)
    for w in words:
        inside[max(0, int(w["s"] / FRAME)):min(m, int(w["e"] / FRAME) + 1)] = True
    speech = float(np.median(db[inside])) if inside.any() else float(np.max(db))
    floor = float(np.percentile(db, 10))
    thr = max(floor + floor_db, speech + rel_db)
    dur = len(x) / SR
    bounds = [(0.0, words[0]["s"])] + [(words[i - 1]["e"], words[i]["s"]) for i in range(1, len(words))] \
        + [(words[-1]["e"], dur)]
    out = []
    for k, (a, b) in enumerate(bounds):
        a2, b2 = a + (dilate if k > 0 else 0.0), b - (dilate if k < len(bounds) - 1 else 0.0)
        fa, fb = int(a2 / FRAME), int(b2 / FRAME)
        if fb - fa < int(min_active / FRAME):
            continue
        act = db[fa:fb] >= thr
        if act.sum() * FRAME < min_active:
            continue
        idx = np.flatnonzero(act)
        s, e = (fa + idx[0]) * FRAME, min((fa + idx[-1] + 1) * FRAME, (fa + idx[0]) * FRAME + max_len)
        out.append({"slot": k, "s": round(s, 3), "e": round(e, 3),
                    "level_db": round(float(np.max(db[fa:fb]) - speech), 1)})
    return out


def clip_of(x, cand, pad_s=0.3):
    """The candidate cut out of `x` and padded with silence on both sides, as the head was trained on."""
    pad = np.zeros(int(pad_s * SR), np.float32)
    return np.concatenate([pad, x[int(cand["s"] * SR):int(cand["e"] * SR)].astype(np.float32), pad])


class NonVerbal:
    def __init__(self, path, device="cuda"):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        self.model = AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=torch.float16).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(str(path))
        cfg = self.model.config
        self.labels = tuple(cfg.labels)
        self.thresholds = dict(cfg.thresholds)
        self.rule = dict(cfg.candidate_rule)
        fh = dict(getattr(cfg, "frame_head", None) or {})
        self.frame_classes = tuple(fh.get("classes") or ())
        self.frame_thresholds = dict(fh.get("thresholds") or {})
        self.fps = fh.get("fps", 10)
        self.device = device

    @classmethod
    def from_pretrained(cls, repo, device="cuda", revision=None):
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo, revision=revision), device)

    def classify(self, clips, batch=32):
        """(N, len(labels)) probabilities for 16 kHz mono clips (at most 30 s; shorter than 0.1 s is padded)."""
        import torch
        clips = [np.asarray(x, np.float32)[: 30 * SR] for x in clips]
        clips = [x if len(x) >= SR // 10 else np.pad(x, (0, SR // 10 - len(x))) for x in clips]
        out = []
        with torch.inference_mode():
            for s in range(0, len(clips), batch):
                part = clips[s:s + batch]
                f = self.fe(part, sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
                n = torch.tensor([max(1, min(1500, int(len(x) / SR * 50))) for x in part], device=self.device)
                out.append(self.model(f, num_frames=n).logits.softmax(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), np.float32)

    def frames(self, x, batch=8):
        """(frames, len(frame_classes)) probabilities every 1/fps s for 16 kHz mono audio of any length (read in
        30 s windows)."""
        import torch
        x = np.asarray(x, np.float32)
        wins = [x[k * 30 * SR:(k + 1) * 30 * SR] for k in range(max(1, int(np.ceil(len(x) / (30 * SR)))))]
        wins = [w if len(w) >= SR // 10 else np.pad(w, (0, SR // 10 - len(w))) for w in wins]
        out = []
        with torch.inference_mode():
            for s in range(0, len(wins), batch):
                part = wins[s:s + batch]
                f = self.fe(part, sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
                p = self.model.frames(f).float().cpu().numpy()
                out += [q[:min(len(q), int(np.ceil(len(w) / SR * self.fps)))] for q, w in zip(p, part)]
        return np.concatenate(out) if out else np.zeros((0, len(self.frame_classes)), np.float32)

    @staticmethod
    def runs(p, t, min_frames=2, join=2):
        """Runs of frames with p >= t, joined across up to `join` quiet frames, at least min_frames long."""
        out = []
        for f in np.flatnonzero(np.asarray(p) >= t):
            if out and f - out[-1][1] <= join:
                out[-1][1] = f + 1
            else:
                out.append([f, f + 1])
        return [(a, b, float(np.max(p[a:b]))) for a, b in out if b - a >= min_frames]

    def decide(self, probs):
        """The class to tag for one candidate, or None: the most probable non-verbal class, if it clears its
        threshold (a class without a threshold is never tagged)."""
        nv = {TO_CLASS[lab]: float(p) for lab, p in zip(self.labels, probs) if lab in TO_CLASS}
        c = max(nv, key=nv.get)
        t = self.thresholds.get(c)
        return c if t is not None and nv[c] >= t else None

    def tag(self, x, words, frames=True):
        """Non-verbal events in one segment: 16 kHz mono audio and its aligned words. -> [{kind, s, e, p, by}], by
        "gap" (with the word slot) or "frames" (with the share of it that is over words)."""
        x = np.asarray(x, np.float32)
        cands = candidates(x, words, **{k: v for k, v in self.rule.items() if k != "pad_s"})
        events = []
        if cands:
            probs = self.classify([clip_of(x, c, self.rule.get("pad_s", 0.3)) for c in cands])
            label_of = {v: k for k, v in TO_CLASS.items()}
            for c, p in zip(cands, probs):
                kind = self.decide(p)
                if kind:
                    events.append({"kind": kind, "s": c["s"], "e": c["e"], "slot": c["slot"], "by": "gap",
                                   "p": round(float(p[self.labels.index(label_of[kind])]), 4)})
        if frames and self.frame_thresholds:
            fp = self.frames(x)
            for cls, t in self.frame_thresholds.items():
                for a, b, peak in self.runs(fp[:, self.frame_classes.index(cls)], t):
                    s, e = a / self.fps, b / self.fps
                    if any(ev["kind"] == cls and ev["s"] <= e + 0.3 and ev["e"] >= s - 0.3 for ev in events):
                        continue
                    over = sum(max(0.0, min(e, w["e"]) - max(s, w["s"])) for w in words) / (e - s)
                    events.append({"kind": cls, "s": round(s, 3), "e": round(e, 3), "p": round(peak, 4),
                                   "by": "frames", "over_speech": round(min(1.0, over), 3)})
        return sorted(events, key=lambda ev: ev["s"])
