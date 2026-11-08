#!/usr/bin/env python3
"""Where a running Melati cluster spends its GPU time: every app replica's batcher stats (/stats) summed per model,
against the audio hours that passed the stage each model serves, so a slow model shows as GPU-seconds per hour of
audio and a backlog as its queue. With --window S the stats are read twice, S seconds apart, against the recordings
whose stage file was written in between: the cost now, not since each server started. RUNS ON THE BOX.

    python scripts/cluster_load.py --endpoints $MELATI_DATA/serve/endpoints.json \\
        --sources $MELATI_DATA/podcast/sources.jsonl --work $MELATI_DATA/podcast/work [--window 600]
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

STAGE_OF = {"front": "sep.json", "tag": "tags.jsonl", "align": "align.jsonl", "mos": "mos.jsonl"}

ap = argparse.ArgumentParser()
ap.add_argument("--endpoints", type=Path, required=True)
ap.add_argument("--sources", type=Path, required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--window", type=float, default=0, help="seconds between two reads of the stats (0: since start)")
args = ap.parse_args()
eps = json.loads(args.endpoints.read_text())


def snapshot():
    out = {}
    for svc in ("front", "tag", "align", "mos"):
        tot = defaultdict(lambda: [0.0, 0, 0])
        for url in eps.get(svc, []):
            try:
                st = json.loads(urlopen(url + "/stats", timeout=10).read())
            except Exception as e:  # noqa: BLE001
                print(f"{svc} {url}: {type(e).__name__}")
                continue
            if svc == "mos":                       # SOTA-MOS's own server reports milliseconds
                st = {"sota-mos": {"gpu_s": st.get("gpu_ms_sum", 0) / 1000, "items": st.get("batched_items", 0),
                                   "queue": 0}}
            elif not all(isinstance(v, dict) for v in st.values()):     # one batcher, reported flat (align)
                st = {svc: st}
            for k, v in st.items():
                tot[k][0] += v.get("gpu_s", 0); tot[k][1] += v.get("items", 0); tot[k][2] = tot[k][2] + v.get("queue", 0)
        out[svc] = tot
    return out


def hours_past(since=None, until=None):
    hours = defaultdict(float)
    for l in args.sources.open():
        d = args.work / json.loads(l)["id"]
        have = set(os.listdir(d)) if d.is_dir() else set()
        if "sep.json" not in have:
            continue
        h = None
        for svc, f in STAGE_OF.items():
            if f not in have:
                continue
            if since is not None and not (since <= os.stat(d / f).st_mtime < until):
                continue
            h = h if h is not None else json.loads((d / "sep.json").read_text())["duration_s"] / 3600
            hours[svc] += h
    return hours


a = snapshot() if args.window else None
t0 = time.time()
if args.window:
    time.sleep(args.window)
b = snapshot()
hours = hours_past(t0, time.time()) if args.window else hours_past()
for svc in ("front", "tag", "align", "mos"):
    h = hours[svc]
    print(f"{svc}: {len(eps.get(svc, []))} replicas, {h:.2f} h of audio past the stage"
          + (f" in {args.window:.0f} s" if args.window else ""))
    for k, (g, n, q) in sorted(b[svc].items(), key=lambda kv: -kv[1][0]):
        if a:
            g, n = g - a[svc][k][0], n - a[svc][k][1]
        print(f"  {k:24s} {g:8.0f} GPU-s {n:7d} items  queue {q:5d}  {g / max(h, 1e-9):7.0f} GPU-s per audio hour")
