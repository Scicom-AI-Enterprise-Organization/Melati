#!/usr/bin/env python3
"""How far a Melati run has got: recordings past each stage, the audio hours finished, failures, and the rate since
the previous call (kept in <work>/../progress.jsonl), with the time left at that rate. RUNS ON THE BOX.

    python scripts/full_progress.py --sources $MELATI_DATA/podcast/sources.jsonl --work $MELATI_DATA/podcast/work
"""
import argparse
import json
import os
import time
from pathlib import Path

STAGES = (("front", "sep.json"), ("diar", "diar.json"), ("seg", "segments.jsonl"), ("asr", "asr.jsonl"),
          ("align", "align.jsonl"), ("quality", "mos.jsonl"), ("tag", "tags.jsonl"), ("sounds", "sounds.jsonl"),
          ("fill", "fill.jsonl"))

ap = argparse.ArgumentParser()
ap.add_argument("--sources", type=Path, required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--block", type=int, default=64, help="as scripts/build_hf_dataset.py cuts its shards")
args = ap.parse_args()
ids = [json.loads(l)["id"] for l in args.sources.open()]
count = {s: 0 for s, _ in STAGES}
done_h, failed, total_bytes, done_bytes = 0.0, 0, 0, 0
size = {json.loads(l)["id"]: json.loads(l).get("bytes", 0) for l in args.sources.open()}
for i in ids:
    d = args.work / i
    have = set(os.listdir(d)) if d.is_dir() else set()
    for s, f in STAGES:
        count[s] += f in have
    total_bytes += size.get(i, 0)
    if "fill.jsonl" in have:
        done_h += json.loads((d / "sep.json").read_text())["duration_s"] / 3600
        done_bytes += size.get(i, 0)
    elif "error.txt" in have and size.get(i, 1024) >= 1024:      # not the source's empty files, left out of the run
        failed += 1
now = time.time()
log = args.work.parent / "progress.jsonl"
prev = [json.loads(l) for l in log.open()] if log.exists() else []
row = {"t": now, "done_h": round(done_h, 2), "done": count["fill"], "failed_now": failed, "stages": count,
       "done_bytes": done_bytes}
with log.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"{time.strftime('%H:%M')} | " + " ".join(f"{s} {count[s]}" for s, _ in STAGES) + f" | of {len(ids)}")
print(f"finished {count['fill']} recordings, {done_h:.1f} h of audio ({100 * done_bytes / max(total_bytes, 1):.1f}% "
      f"of the source by size); {failed} with an error and not finished")
fin = [(args.work / i / "fill.jsonl").exists() or size.get(i, 1024) < 1024     # an empty source file settles too
       for i in ids]                                                              # (scripts/build_hf_dataset.py)
blocks = [all(fin[b:b + args.block]) for b in range(0, len(ids), args.block)]
print(f"blocks of {args.block} complete: {sum(blocks)} of {len(blocks)}")
if prev:
    p = prev[0] if len(prev) < 2 else prev[-1]
    dt = now - p["t"]
    rate_h = (done_h - p["done_h"]) / dt * 3600          # audio hours per wall hour since the previous call
    rate_b = (done_bytes - p.get("done_bytes", 0)) / dt
    left_s = (total_bytes - done_bytes) / rate_b if rate_b > 0 else float("inf")
    print(f"since {time.strftime('%H:%M', time.localtime(p['t']))}: {rate_h:.0f}x realtime, "
          f"{rate_b * 3600 / 2 ** 30:.2f} GiB of source an hour; at that rate {left_s / 3600:.1f} h left")
