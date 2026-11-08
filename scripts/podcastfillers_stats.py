#!/usr/bin/env python3
"""Events per split and label in PodcastFillers.csv: what a non-verbal head can train on. RUNS ON THE BOX.

    python scripts/podcastfillers_stats.py /path/to/PodcastFillers.csv
"""
import csv, sys
from collections import Counter

c, cons, dur = Counter(), Counter(), Counter()
with open(sys.argv[1], encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        k = (r["episode_split_subset"], r["label_full_vocab"])
        c[k] += 1
        cons[(r["label_full_vocab"], r["label_consolidated_vocab"])] += 1
        dur[r["label_full_vocab"]] += float(r["event_end_inepisode"]) - float(r["event_start_inepisode"])
labels = sorted({l for _, l in c}, key=lambda l: -sum(v for (s, x), v in c.items() if x == l))
splits = ["train", "validation", "test"]
print(f"{'label':14s}" + "".join(f"{s:>12s}" for s in splits) + f"{'mean s':>9s}  consolidated as")
for l in labels:
    n = sum(c[(s, l)] for s in splits)
    print(f"{l:14s}" + "".join(f"{c[(s, l)]:12,d}" for s in splits) + f"{dur[l] / max(n, 1):9.2f}  "
          + ", ".join(f"{b} {v}" for (a, b), v in cons.items() if a == l))
