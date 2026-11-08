#!/usr/bin/env python3
"""Non-verbal events Melati exported, by kind, for one or more work directories. RUNS ON THE BOX.
With --into and --key, the counts of the (single) work directory are also kept under that key of a JSON file
(results/compare/pilot_events.json: the pilot's tags per labeller, which the paper reads).

    python scripts/count_events.py $MELATI_DATA/pilot/work_nvwhisper $MELATI_DATA/pilot/work_nvwhisper3
    python scripts/count_events.py $MELATI_DATA/pilot/work_nvrel3 \\
        --into $MELATI_DATA/results/compare/pilot_events.json --key head
"""
import argparse, json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("work", nargs="+", type=Path)
ap.add_argument("--into", type=Path, default=None)
ap.add_argument("--key", default=None)
args = ap.parse_args()
for work in args.work:
    kinds, segs, hours = Counter(), 0, 0.0
    for f in sorted(work.glob("*/export.jsonl")):
        for line in f.open(encoding="utf-8"):
            row = json.loads(line)
            segs += 1
            hours += (row["end"] - row["start"]) / 3600
            ann = json.loads(row["annotation"]) if isinstance(row.get("annotation"), str) else row.get("annotation") or {}
            kinds.update(e.get("kind") for e in ann.get("events") or [])
    res = {"work": str(work), "segments": segs, "hours": round(hours, 2), "events": dict(kinds.most_common()),
           "per_hour": {k: round(v / max(hours, 1e-9), 1) for k, v in kinds.most_common()}}
    print(json.dumps(res))
    if args.into and args.key:
        d = json.loads(args.into.read_text()) if args.into.exists() else {}
        d[args.key] = res
        args.into.write_text(json.dumps(d, indent=1))
