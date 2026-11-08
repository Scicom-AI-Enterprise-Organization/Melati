#!/usr/bin/env python3
"""The recordings of a Melati run that failed and have not finished since: how long each is, the stage it stopped
at and its error, against the durations of the recordings that did finish, so a failure that tracks length shows
as one. A finished recording's diarization time is its diar.json's mtime less its sep.json's (the stage before it),
so it includes any wait for a free call. RUNS ON THE BOX.

    python scripts/full_failures.py --sources $MELATI_DATA/podcast/sources.jsonl --work $MELATI_DATA/podcast/work
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

BANDS = (0, 10, 20, 30, 45, 60, 80, 120, 10 ** 6)          # minutes
STAGES = (("front", "sep.json"), ("diar", "diar.json"), ("seg", "segments.jsonl"), ("asr", "asr.jsonl"),
          ("align", "align.jsonl"), ("quality", "mos.jsonl"), ("tag", "tags.jsonl"), ("fill", "fill.jsonl"))

ap = argparse.ArgumentParser()
ap.add_argument("--sources", type=Path, required=True)
ap.add_argument("--work", type=Path, required=True)
args = ap.parse_args()


def band(m):
    k = next(i for i in range(len(BANDS) - 1) if m < BANDS[i + 1])
    return f"{BANDS[k]}-{BANDS[k + 1]}" if BANDS[k + 1] < 10 ** 6 else f"{BANDS[k]}+"


done, failed, diar_mode, diar_min = Counter(), [], Counter(), defaultdict(list)
for l in args.sources.open():
    i = json.loads(l)["id"]
    d = args.work / i
    have = set(os.listdir(d)) if d.is_dir() else set()
    if "sep.json" not in have:
        continue
    m = json.loads((d / "sep.json").read_text())["duration_s"] / 60
    if "fill.jsonl" in have:
        done[band(m)] += 1
        mode = json.loads((d / "diar.json").read_text()).get("mode")
        diar_mode[(band(m), mode)] += 1
        diar_min[(band(m), mode)].append((os.stat(d / "diar.json").st_mtime - os.stat(d / "sep.json").st_mtime) / 60)
    elif "error.txt" in have:
        stage = next(s for s, f in STAGES if f not in have)
        failed.append((m, i, stage, (d / "error.txt").read_text().strip().splitlines()[-1][:110]))
fail_by = Counter(band(m) for m, *_ in failed)


def spread(xs):
    xs = sorted(xs)
    return f"{xs[len(xs) // 2]:5.1f} / {xs[-1]:5.1f}" if xs else "    -  /    - "


print("minutes   finished  (one pass / windows)  failed   diar minutes, one pass (median / max)   windows")
for k in range(len(BANDS) - 1):
    b = band(BANDS[k])
    print(f"{b:>8} {done[b]:9d}  ({diar_mode[(b, 'one')]:5d} / {diar_mode[(b, 'windows')]:5d}) {fail_by[b]:9d}   "
          f"{spread(diar_min[(b, 'one')]):>36}   {spread(diar_min[(b, 'windows')])}")
print(f"{len(failed)} failed:")
for m, i, stage, err in sorted(failed):
    print(f"  {i} {m:6.1f} min  {stage:6s} {err}")
