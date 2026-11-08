#!/usr/bin/env python3
"""How much of AudioSet the box holds: for each split of agkphysics/AudioSet, its shards and bytes on the Hub against
the shards audioset/data.py has decoded into the cache (and their clips), so the size of fetching the rest is a
number, not a guess. RUNS ON THE BOX.

    python audioset/data_coverage.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT  # noqa: E402

files = [f for f in HfApi().list_repo_tree("agkphysics/AudioSet", repo_type="dataset", recursive=True)
         if getattr(f, "size", None) and f.path.endswith(".parquet")]
out = {}
for split in sorted({f.path.split("/")[-2] for f in files}):
    hub = {Path(f.path).stem: f.size for f in files if f.path.split("/")[-2] == split}
    cache = ROOT / "cache" / split
    have = {p.stem for p in cache.glob("*.npz")} if cache.is_dir() else set()
    clips = 0
    for stem in have:
        try:
            clips += len(np.load(cache / f"{stem}.npz")["ids"])
        except Exception:
            pass
    one = (ROOT / "cache" / f"{split}_meta.json")
    if one.exists():                                         # a split cached as one array (eval, bal_train)
        clips = json.loads(one.read_text()).get("clips", clips)
        have = set(hub) if not have else have
    left = {s: b for s, b in hub.items() if s not in have}
    out[split] = {"hub_shards": len(hub), "hub_gb": round(sum(hub.values()) / 1e9, 1), "cached_shards": len(have & set(hub)),
                  "cached_clips": clips, "left_shards": len(left), "left_gb": round(sum(left.values()) / 1e9, 1)}
print(json.dumps(out, indent=1))
