#!/usr/bin/env python3
"""The teachers' probabilities for every AudioSet training clip, shard by shard as deploy/box_audioset_fetch.sh lands
them, and each shard's clips cached at 16 kHz for training: cache/<split>/<shard>.npy (float16, 10 s rows, a short
clip padded with zeros) and <shard>.npz (ids, lengths, the 527-column labels, and each teacher's probabilities as
float16). A shard already done is skipped; with --follow it waits for the fetch to land more until every shard of the
split is done. Teachers run in bf16 (audioset/zoo.py: at most 0.1 mAP point from fp32). RUNS ON THE BOX, in .venv-sed.

    CUDA_VISIBLE_DEVICES=0 python audioset/targets.py --split unbal_train --slice 0/5 --follow
"""
import os
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, _shard, classes  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

TEACHERS = ("ced-base", "eat-large", "sslam")      # the best average on the evaluation set (results/audioset)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="unbal_train", help="the cache folder's name, and agk's split by default")
    ap.add_argument("--src", type=Path, default=None, help="the parquet folder, if not agk's split (e.g. the strong "
                    f"subset: {MELATI_DATA}/audioset/strong/data/train)")
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--follow", action="store_true", help="wait for more shards until the split is complete")
    ap.add_argument("--expect", type=int, default=None, help="shards the split has (default: from the repo listing)")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    i, n = map(int, a.slice.split("/"))
    src = a.src or ROOT / "agk" / "data" / a.split
    out = ROOT / "cache" / a.split
    out.mkdir(parents=True, exist_ok=True)
    expect = a.expect or {"unbal_train": 870, "bal_train": 38, "eval": 35}.get(a.split)
    mid = {m: k for k, m, _ in classes()}
    from zoo import load as load_model
    models = {t: load_model(t) for t in TEACHERS}
    done_any, fails = 0, {}
    while True:
        # a slice by the name's checksum, not its place in the folder: places shift while the fetch still lands files,
        # which once left a file in no slice at all
        mine = [s for s in sorted(src.glob("*.parquet")) if zlib.crc32(s.name.encode()) % n == i]
        todo = [s for s in mine if not (out / f"{s.stem}.npz").exists() and fails.get(s.name, 0) < 3]
        for s in todo:
            t0 = time.time()
            try:
                ids, W, lens, labs, rows, strong = _shard(s)
            except Exception as e:                          # corrupt: tried three times, then listed for a refetch
                fails[s.name] = fails.get(s.name, 0) + 1
                print(f"skip {s.name} ({fails[s.name]}): {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            Y = np.zeros((len(ids), len(mid)), np.bool_)
            for k, y in enumerate(labs):
                for m in y:
                    if m in mid:
                        Y[k, mid[m]] = True
            probs = {}
            for t, m in models.items():
                probs[t] = np.concatenate([m([w.astype(np.float32) for w in W[j:j + a.batch]]).astype(np.float16)
                                           for j in range(0, len(W), a.batch)]) if len(W) else np.zeros((0, 527))
            np.save(out / f"{s.stem}.npy", W)
            extra = {"strong": np.array([json.dumps(st) for st in strong])} if strong is not None else {}
            np.savez(out / f"{s.stem}.npz", ids=np.array(ids), lens=np.array(lens), labels=Y,
                     **{f"p_{t}": p for t, p in probs.items()}, **extra)
            done_any += 1
            print(f"{s.name}: {len(ids)} of {rows} clips, {time.time() - t0:.0f} s", flush=True)
        have = len(list(out.glob("*.npz")))
        settled = [s for s in mine if (out / f"{s.stem}.npz").exists() or fails.get(s.name, 0) >= 3]
        landed = len(list(src.glob("*.parquet")))
        if not a.follow or (expect and landed >= expect and len(settled) == len(mine)):  # every file in, all of mine done
            break
        time.sleep(120)
    bad = sorted(k for k, v in fails.items() if v >= 3)
    print(json.dumps({"split": a.split, "slice": a.slice, "shards_done_here": done_any, "shards_cached": have,
                      "corrupt": bad}))
    print("TARGETS_DONE", flush=True)


if __name__ == "__main__":
    main()
