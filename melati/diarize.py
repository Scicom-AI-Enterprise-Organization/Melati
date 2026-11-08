"""Who-Spoke-When-Over-the-Phone's text format, and the windowing for long recordings.

The model (Scicom-intl/Who-Spoke-When-Over-the-Phone-Whisper-Large-V3-Merged) writes turns as
`[start][Sxx][end]` for up to an hour of audio in one pass and follows up to sixteen voices. A pass
that hits its token budget (24 tokens per second of audio, the model card's sizing), whose
timestamps run past the end, or that gets no answer in two half-hour tries has failed; that
recording, and any over 45 minutes, is decoded in ten-minute windows overlapping by a minute. The
model takes 80 minutes, but a pass over 45 holds over a quarter of a shared server's KV cache while
it decodes at tens of tokens a second, and a few of them stop the server admitting anything else
(the full run, 2026-10-26: 6x realtime, every recording in flight waiting on diarization). Windows
are not linked into recording-wide speakers here (the release's diarize.py links them with WeSpeaker
embeddings): a windowed recording's speakers are named per window, `w<k>S<j>`, which costs speaker
identity across windows and never purity within a turn.
"""
from __future__ import annotations

import math
import re

PROMPT = ("Diarize the audio. For each segment, output the start timestamp, the speaker ID ([S01], [S02], [S03], ...) "
          "and the end timestamp, without transcribing the speech.")
TURN = re.compile(r"\[(\d+(?:\.\d+)?)\]\[S(\d+)\](?:[^\[]*)\[(\d+(?:\.\d+)?)\]")
ONE_PASS_MAX_S = 45 * 60          # 80 until 2026-10-26 08:12; see above
WINDOW_S, OVERLAP_S = 600.0, 60.0


def turns_of(text: str):
    return [(float(s), float(e), int(k)) for s, k, e in TURN.findall(text)]


def failed(text: str, budget_hit: bool, duration: float) -> bool:
    return budget_hit or max((e for _, e, _ in turns_of(text)), default=0.0) > duration + 30


def plan(duration: float, window: float = WINDOW_S, overlap: float = OVERLAP_S):
    if duration <= window:
        return [(0.0, duration)]
    n = math.ceil((duration - window) / (window - overlap)) + 1
    step = (duration - window) / (n - 1)
    return [(k * step, k * step + window) for k in range(n)]


def stitch(windows, texts):
    """Turns of overlapping windows; each window owns the span between its overlaps' midpoints."""
    K = len(windows)
    mids = [windows[0][0]] + [(windows[k + 1][0] + windows[k][1]) / 2 for k in range(K - 1)] + [windows[-1][1]]
    out = []
    for k, ((ws, we), text) in enumerate(zip(windows, texts)):
        for s, e, spk in turns_of(text):
            s, e = max(s + ws, mids[k]), min(e + ws, we, mids[k + 1])
            if e > s:
                out.append([round(s, 3), round(e, 3), f"w{k}S{spk:02d}"])
    return sorted(out)
