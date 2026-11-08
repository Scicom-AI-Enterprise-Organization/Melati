"""Voiced audio that no aligned word accounts for: where fillers and missed words hide.

Whisper writes clean text. A filled pause ("erm", "emm", "aaa") is usually not in its transcript,
and a word the transcript does not have cannot be aligned. So the aligner leaves a GAP over it,
and the gap is not silent. This module finds those gaps: the intervals between consecutive
aligned words (and before the first / after the last word of a segment) where the speech stem is
voiced for long enough to be a sound someone made. What each one IS is decided later: an event the
non-verbal detector already explains, a filler (the gap transcribed on its own reads as one), or a
word the transcript dropped (it reads as a word).

Studio reads show the same shape (voiced audio outside every dilated word span), and it is the
reason a filler in Melati carries real times.
"""
from __future__ import annotations

import numpy as np

FRAME_S = 0.01


def frame_db(x: np.ndarray, sr: int, frame_s: float = FRAME_S) -> np.ndarray:
    n = int(frame_s * sr)
    m = len(x) // n
    if m == 0:
        return np.zeros(0)
    f = x[: m * n].reshape(m, n).astype(np.float64)
    return 10 * np.log10(np.mean(f * f, axis=1) + 1e-12)


def voiced_gaps(words, x: np.ndarray, sr: int, min_gap=0.15, min_voiced=0.10, rel_db=-18.0, dilate=0.03):
    """[{s, e, voiced_s, level_db}] in segment time. A frame is voiced when it is within `rel_db`
    of the segment's speech level (the median frame level inside aligned words); words are
    dilated by `dilate` so their own onsets and releases are not counted."""
    db = frame_db(x, sr)
    if len(db) == 0 or not words:
        return []
    dur = len(x) / sr
    inside = np.zeros(len(db), bool)
    for w in words:
        a, b = int(w["s"] / FRAME_S), int(w["e"] / FRAME_S) + 1
        inside[max(0, a):min(len(db), b)] = True
    speech_level = float(np.median(db[inside])) if inside.any() else float(np.max(db))
    thr = speech_level + rel_db
    bounds = [(0.0, words[0]["s"])] + [(words[i - 1]["e"], words[i]["s"]) for i in range(1, len(words))] \
        + [(words[-1]["e"], dur)]
    out = []
    for k, (a, b) in enumerate(bounds):
        a2, b2 = a + (dilate if k > 0 else 0.0), b - (dilate if k < len(bounds) - 1 else 0.0)
        if b2 - a2 < min_gap:
            continue
        fa, fb = int(a2 / FRAME_S), int(b2 / FRAME_S)
        v = db[fa:fb] >= thr
        if v.sum() * FRAME_S < min_voiced:
            continue
        idx = np.flatnonzero(v)
        vs, ve = (fa + idx[0]) * FRAME_S, (fa + idx[-1] + 1) * FRAME_S
        out.append({"s": round(vs, 3), "e": round(ve, 3), "voiced_s": round(float(v.sum() * FRAME_S), 3),
                    "level_db": round(float(np.max(db[fa:fb]) - speech_level), 1), "slot": k})
    return out


def syllables(word: str) -> int:
    """A rough syllable count: vowel groups for alphabetic scripts, one per CJK character."""
    import re
    cjk = len(re.findall(r"[㐀-鿿぀-ヿ가-힯]", word))
    if cjk:
        return cjk
    groups = re.findall(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿ]+", word.lower())
    return max(1, len(groups))


def rate_spans(words, fast=1.6, slow=0.6, min_words=5, min_s=1.5, min_segment_words=12):
    """Delivery spans from speaking rate: runs of at least `min_words` words (and `min_s` seconds)
    whose local rate (syllables per second over a 5-word window) is above `fast` x or below `slow` x
    the segment's median, in segments of at least `min_segment_words` words. -> [(label, first, last)].

    The first version (1.35x / 0.7x over 3 words) tagged a third of the VoxConverse dev segments:
    the rate of three words is mostly the noise of syllable counting, not a change of pace. Export
    keeps these spans off until they are checked by ear (export_config `rate_spans`)."""
    if len(words) < max(min_segment_words, min_words + 2):
        return []
    rates = []
    for i in range(len(words)):
        a, b = max(0, i - 2), min(len(words) - 1, i + 2)
        dur = max(words[b]["e"] - words[a]["s"], 1e-3)
        rates.append(sum(syllables(w["w"]) for w in words[a:b + 1]) / dur)
    med = float(np.median(rates))
    if med <= 0:
        return []
    lab = ["fast" if r > fast * med else "slow" if r < slow * med else None for r in rates]
    out, i = [], 0
    while i < len(lab):
        if lab[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        if j - i + 1 >= min_words and words[j]["e"] - words[i]["s"] >= min_s:
            out.append((lab[i], i, j))
        i = j + 1
    return out
