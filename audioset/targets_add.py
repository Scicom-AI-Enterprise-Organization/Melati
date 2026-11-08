#!/usr/bin/env python3
"""More teachers for clips audioset/targets.py has already cached: each one's probabilities for every row of a shard,
in a file of its own beside the shard (<shard>.<name>.npy, float16, the shard's row order), so the shard's audio and
its first teachers are never rewritten. audioset/pool.py --teachers then averages them with the rest. A shard that
has the file is skipped. Run in bf16, as the first teachers were. RUNS ON THE BOX, in .venv-sed, one process a GPU.

    CUDA_VISIBLE_DEVICES=0 python audioset/targets_add.py --sources strong_train bal_train unbal_train --slice 0/5 \\
        --teacher v4=melati:$MELATI_DATA/models/audioset/frames1_fw0/clip.pt
"""
import os
import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help=f"cache folders under {MELATI_DATA}/audioset/cache")
    ap.add_argument("--teacher", action="append", required=True, metavar="NAME=MODEL",
                    help="a short name for the file and an audioset/zoo.py model (melati:<checkpoint> included)")
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    i, n = map(int, a.slice.split("/"))
    from zoo import load as load_model
    teachers = dict(t.split("=", 1) for t in a.teacher)
    models = {name: load_model(m) for name, m in teachers.items()}
    shards = [f for src in a.sources for f in sorted((ROOT / "cache" / src).glob("*.npz"))
              if zlib.crc32(f"{src}/{f.name}".encode()) % n == i]
    t0, clips, done = time.time(), 0, 0
    for f in shards:
        want = {name: f.with_name(f"{f.stem}.{name}.npy") for name in models}
        want = {name: p for name, p in want.items() if not p.exists()}
        if not want:
            continue
        try:
            W = np.load(f.with_suffix(".npy"), mmap_mode="r")
        except Exception as e:                       # a cache written when its job was stopped; pool.py leaves it out
            print(f"skip {f.parent.name}/{f.name}: {type(e).__name__}", flush=True)
            continue
        for name, path in want.items():
            m = models[name]
            P = np.concatenate([m([np.asarray(w, np.float32) for w in W[j:j + a.batch]]).astype(np.float16)
                                for j in range(0, len(W), a.batch)]) if len(W) else np.zeros((0, 527), np.float16)
            tmp = path.with_name(path.name + ".part")
            with tmp.open("wb") as fh:
                np.save(fh, P)
            tmp.replace(path)
        clips += len(W)
        done += 1
        if done % 20 == 0:
            print(f"[add] {done} shards, {clips:,} clips, {clips / (time.time() - t0):.0f} clips/s", flush=True)
    print(f"[add] slice {a.slice}: {done} shards, {clips:,} clips in {time.time() - t0:.0f} s", flush=True)
    print("TARGETS_ADD_DONE", flush=True)


if __name__ == "__main__":
    main()
