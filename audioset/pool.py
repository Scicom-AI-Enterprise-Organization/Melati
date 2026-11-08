#!/usr/bin/env python3
"""The clips a student trains on, gathered from audioset/targets.py's shard caches into one 16 kHz array (float16, 10 s
rows) and one table: ids, lengths, AudioSet's 527-column labels, the teachers' average probabilities (audioset/targets.py
TEACHERS, each in bf16) and, for the strongly labelled clips, every timed event. Any clip whose id is in the evaluation
set is left out, so nothing a score is read on is trained on, and a clip in two sources is kept once. RUNS ON THE BOX.

    python audioset/pool.py --sources strong_train bal_train --name clean
    python audioset/pool.py --sources strong_train bal_train unbal_train --name big --index-only
"""
import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CLIP, ROOT, load  # noqa: E402
from targets import TEACHERS  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help=f"cache folders under {MELATI_DATA}/audioset/cache")
    ap.add_argument("--name", required=True)
    ap.add_argument("--exclude", default="eval", help="the cached split whose ids may not be trained on")
    ap.add_argument("--teachers", nargs="+", default=list(TEACHERS),
                    help="whose probabilities to average: a teacher audioset/targets.py cached in the shard, or one "
                         "audioset/targets_add.py added beside it (<shard>.<name>.npy)")
    ap.add_argument("--index-only", action="store_true",
                    help="no copy of the audio: the pool points into the shard caches (for a pool too big to copy)")
    a = ap.parse_args()
    _, held, _, _ = load(a.exclude)
    held = set(held.tolist())
    parts = [(src, f) for src in a.sources for f in sorted((ROOT / "cache" / src).glob("*.npz"))]
    seen, keep = set(), []                                    # (shard npz, rows kept)
    dropped = {"in_eval": 0, "duplicate": 0, "unreadable_shards": 0}
    for src, f in parts:
        try:                                                  # a cache written when its job was stopped: left out
            ids = np.load(f)["ids"]
            np.load(f.with_suffix(".npy"), mmap_mode="r")
        except Exception:
            dropped["unreadable_shards"] += 1
            continue
        rows = []
        for k, vid in enumerate(ids.tolist()):
            if vid in held:
                dropped["in_eval"] += 1
            elif vid in seen:
                dropped["duplicate"] += 1
            else:
                seen.add(vid)
                rows.append(k)
        keep.append((f, rows))
    n = sum(len(r) for _, r in keep)
    W = None if a.index_only else np.lib.format.open_memmap(ROOT / "cache" / f"pool_{a.name}_wav16k.npy", mode="w+",
                                                           dtype=np.float16, shape=(n, CLIP))
    ids, lens, labels, probs, strong, src_of, where, files = [], [], [], [], [], [], [], []
    at = 0
    for f, rows in keep:
        if not rows:
            continue
        z = np.load(f)
        if W is not None:
            w = np.load(f.with_suffix(".npy"), mmap_mode="r")
            W[at:at + len(rows)] = w[rows]
        else:                                                  # an index into the shard caches: nothing is copied
            where += [(len(files), r) for r in rows]
            files.append(str(f.with_suffix(".npy")))
        at += len(rows)
        ids += z["ids"][rows].tolist(); lens += z["lens"][rows].tolist(); labels.append(z["labels"][rows])
        probs.append(np.mean([(z[f"p_{t}"] if f"p_{t}" in z.files else np.load(f.with_name(f"{f.stem}.{t}.npy")))
                              [rows].astype(np.float32) for t in a.teachers], 0).astype(np.float16))
        strong += z["strong"][rows].tolist() if "strong" in z.files else ["null"] * len(rows)
        src_of += [f.parent.name] * len(rows)
    if W is not None:
        W.flush()
    extra = {"files": np.array(files), "where": np.array(where, np.int64)} if W is None else {}
    np.savez(ROOT / "cache" / f"pool_{a.name}_meta.npz", ids=np.array(ids), lens=np.array(lens),
             labels=np.concatenate(labels), teacher=np.concatenate(probs), strong=np.array(strong),
             source=np.array(src_of), **extra)
    meta = {"name": a.name, "sources": a.sources, "teachers": list(a.teachers), "clips": n, "index_only": a.index_only,
            "hours": round(sum(lens) / 16000 / 3600, 1), "dropped": dropped,
            "with_timed_events": int(sum(s != "null" for s in strong)),
            "by_source": {s: src_of.count(s) for s in sorted(set(src_of))}}
    (ROOT / "cache" / f"pool_{a.name}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
