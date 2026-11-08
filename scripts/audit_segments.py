#!/usr/bin/env python3
"""Segments that a segmentation defect damaged: negative length, speech shared with a neighbour, or cut
short of their own speech by the padding clamp. Reads <work>/<rec>/segments.jsonl only.

    python scripts/audit_segments.py $MELATI_DATA/pilot/work $MELATI_DATA/dev/vox/work3
"""
import json, sys
from pathlib import Path

for root in map(Path, sys.argv[1:]):
    recs = n = neg = shared = short = hit = 0
    for f in sorted(root.glob("*/segments.jsonl")):
        segs = [json.loads(l) for l in f.open()]
        bad = 0
        for k, s in enumerate(segs):
            n += 1
            neg += s["end"] <= s["start"]
            shared += k > 0 and s["speech_start"] < segs[k - 1]["speech_end"]
            short += s["end"] < s["speech_end"] - 0.01
            bad += (s["end"] <= s["start"]) or (s["end"] < s["speech_end"] - 0.01)
        recs += 1
        hit += bad > 0
    print(f"{root}: {recs} recordings, {n} segments; negative {neg}, sharing speech {shared}, "
          f"cut short of their own speech {short}; recordings affected {hit}")
