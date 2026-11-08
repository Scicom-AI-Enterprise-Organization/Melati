"""Non-verbal vocalisations: frame probabilities -> events -> Drama-3 point tags and delivery scopes.

Detector: Speech Event Detector (theboringai-work/surround-sound, arXiv 2610.14232), a 5.8 M
parameter conv-Transformer that scores 54 classes every 40 ms, 20 of them sounds a speaker makes.
Its own numbers are modest (event F1 0.33 macro; laugh 0.63, sigh 0.78, breath 0.61), and it
locates nothing shorter than ~0.2 s inside speech, so it is a CANDIDATE generator here: every
event it proposes is kept with its probability, verified downstream, and the threshold that
decides what becomes a tag is chosen on labelled conversational speech (scripts/eval_nonverbal.py),
not taken from the model card.

It reads the vocal stem (speech + effects): a separator trained on Divide and Remaster can file a
cough under "effects", and the speech stem alone would then hide it.

Its classes map onto Melati's tags (melati.tags.NONVERBAL); `whisper` and `shout` are not sounds
but ways of speaking, so they become delivery scopes over the words they cover.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TO_TAG = {"laugh": "laughs", "sigh": "sighs", "gasp": "gasps", "cough": "coughs", "sneeze": "sneezes",
          "sniff": "sniffs", "throat_clear": "clears throat", "breath": "breathes", "cry": "cries",
          "yawn": "yawns", "scream": "screams", "groan": "groans", "hum": "hums"}
TO_DELIVERY = {"whisper": "whisper", "shout": "shout"}
FRAME_S = 0.04


class SED:
    def __init__(self, repo: str = "theboringai-work/surround-sound", profile: str = "sensitive",
                 device: str = "cuda"):
        import sys
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo)
        sys.path.insert(0, path)
        from sed import Detector                           # the release's own code
        self.det = Detector.from_pretrained(path, profile=profile)
        try:
            self.det.model.to(device)
            self.det.device = device
        except Exception:  # noqa: BLE001 - the release may manage its own device
            pass
        cfg = json.loads((Path(path) / "config.json").read_text())
        self.classes = list(cfg.get("classes") or cfg.get("labels") or [])
        thr = json.loads((Path(path) / ("thresholds_sensitive.json" if profile == "sensitive"
                                          else "thresholds.json")).read_text())
        self.thresholds = thr.get("thresholds", thr) if isinstance(thr, dict) else {}
        dc = Path(path) / "detect_config.json"
        self.hysteresis = json.loads(dc.read_text()).get("hysteresis_ratio", 0.5) if dc.exists() else 0.5

    def frame_probs(self, x16k: np.ndarray):
        times, probs = self.det.frame_probs(x16k)
        return np.asarray(times, np.float32), np.asarray(probs, np.float32)


def events_from_probs(times, probs, classes, thresholds, hysteresis=0.5, min_dur=0.08, merge_gap=0.2,
                      t0=0.0, t1=None, scale=1.0):
    """Hysteresis events per class in [t0, t1): starts when p crosses thr, ends when it falls under
    thr * hysteresis. `scale` multiplies every threshold (scale < 1 = more recall)."""
    t1 = times[-1] + FRAME_S if t1 is None else t1
    sel = (times >= t0) & (times < t1)
    tt, pp = times[sel], probs[sel]
    out = []
    for c, name in enumerate(classes):
        if name not in TO_TAG and name not in TO_DELIVERY:
            continue
        hi = float(thresholds.get(name, 0.5)) * scale
        lo = hi * hysteresis
        on, start, peak = False, 0.0, 0.0
        for t, p in zip(tt, pp[:, c]):
            if not on and p >= hi:
                on, start, peak = True, float(t), float(p)
            elif on:
                peak = max(peak, float(p))
                if p < lo:
                    out.append([name, start, float(t), peak]); on = False
        if on:
            out.append([name, start, float(tt[-1]) + FRAME_S, peak])
    out.sort(key=lambda e: (e[0], e[1]))
    merged = []
    for e in out:
        if merged and merged[-1][0] == e[0] and e[1] - merged[-1][2] <= merge_gap:
            merged[-1][2], merged[-1][3] = e[2], max(merged[-1][3], e[3])
        else:
            merged.append(e)
    return [{"label": n, "s": round(s - t0, 3), "e": round(e - t0, 3), "p": round(p, 3)}
            for n, s, e, p in merged if e - s >= min_dur]


def events_from_tracks(tracks: dict, fps: int, t0: float, t1: float, thresholds: dict | None = None,
                       default: float = 0.10, hysteresis: float = 0.5, min_dur: float = 0.08,
                       merge_gap: float = 0.2):
    """Candidate events in [t0, t1) from the stored frame tracks of melati.sed (PANNs groups):
    hysteresis per group (on at thr, off under thr * hysteresis), gaps under merge_gap joined,
    shorter than min_dur dropped. Times are relative to t0; `p` is the peak probability, which
    export compares with its per-class floor. Only groups that map to a tag or a delivery span."""
    thresholds = thresholds or {}
    a, b = int(t0 * fps), int(np.ceil(t1 * fps))
    out = []
    for name, track in tracks.items():
        if name not in TO_TAG and name not in TO_DELIVERY:
            continue
        v = np.asarray(track[a:b], dtype=np.float32)
        if len(v) == 0:
            continue
        hi = float(thresholds.get(name, default))
        lo = hi * hysteresis
        on, start, peak = False, 0, 0.0
        spans = []
        for i, p in enumerate(v):
            if not on and p >= hi:
                on, start, peak = True, i, float(p)
            elif on:
                peak = max(peak, float(p))
                if p < lo:
                    spans.append([start, i, peak]); on = False
        if on:
            spans.append([start, len(v), peak])
        merged = []
        for s_, e_, p_ in spans:
            if merged and (s_ - merged[-1][1]) / fps <= merge_gap:
                merged[-1][1], merged[-1][2] = e_, max(merged[-1][2], p_)
            else:
                merged.append([s_, e_, p_])
        out += [{"label": name, "s": round(s_ / fps, 3), "e": round(e_ / fps, 3), "p": round(p_, 3)}
                for s_, e_, p_ in merged if (e_ - s_) / fps >= min_dur]
    return sorted(out, key=lambda e: e["s"])
