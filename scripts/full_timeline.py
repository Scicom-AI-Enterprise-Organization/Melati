#!/usr/bin/env python3
"""The full run's progress over time, from its own files: when each recording finished its last stage (fill.jsonl's
modification time) and how long it is (sep.json), as cumulative hours of audio against wall-clock hours, in 10-minute
steps with the realtime factor of each step. The paper's throughput figure reads it. RUNS ON THE BOX.

    python scripts/full_timeline.py --sources $MELATI_DATA/podcast/sources.jsonl \\
        --work $MELATI_DATA/podcast/work --out $MELATI_DATA/results/dataset/timeline.json
"""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--sources", type=Path, required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--step-min", type=float, default=10)
ap.add_argument("--event", action="append", default=[], metavar="'YYYY-MM-DDTHH:MM label'",
                help="a change made while the run went, at its local time (from the deploy logs), to mark on the plot")
ap.add_argument("--events", type=Path, default=None, help="a file of such lines (deploy/full_run_events.txt)")
args = ap.parse_args()
if args.events:
    args.event += [l.strip() for l in args.events.read_text().splitlines() if l.strip() and not l.startswith("#")]
done, began = [], []
for l in args.sources.open():
    d = args.work / json.loads(l)["id"]
    try:
        t = os.stat(d / "fill.jsonl").st_mtime
    except FileNotFoundError:
        continue
    done.append((t, json.loads((d / "sep.json").read_text())["duration_s"] / 3600))
    began.append(os.stat(d / "sep.json").st_mtime)        # its first stage's file: the run had started by then
done.sort()
start = min(began) if began else 0
step = args.step_min * 60
points, h, k = [], 0.0, 0
end = done[-1][0] if done else 0
edge = start
while edge <= end + step:
    prev = h
    while k < len(done) and done[k][0] < edge:
        h += done[k][1]
        k += 1
    points.append({"wall_h": round((edge - start) / 3600, 3), "audio_h": round(h, 2),
                   "x_realtime": round((h - prev) / (step / 3600), 1) if points else None})
    edge += step
import time  # noqa: E402
events = []
for e in args.event:
    when, label = e.split(" ", 1)
    events.append({"wall_h": round((time.mktime(time.strptime(when, "%Y-%m-%dT%H:%M")) - start) / 3600, 3),
                   "label": label})
out = {"recordings": len(done), "events": events, "audio_h": round(h, 1), "wall_h": round((end - start) / 3600, 2),
       "x_realtime": round(h / max((end - start) / 3600, 1e-9), 1), "start": start, "step_min": args.step_min,
       "points": points}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "points"}))
