#!/usr/bin/env python3
"""Count the values of one column across a dataset config's parquet files, reading only that column
(HfFileSystem + pyarrow column projection: a few MB for millions of rows), plus summed durations.

    python asr/scripts/count_hf_column.py Scicom-intl/TTS-Clean44k labeled44k --column lang
"""
import argparse, os
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("config")
    ap.add_argument("--column", default="lang")
    ap.add_argument("--also", nargs="*", default=["source"], help="columns to cross-tab against --column")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    tok = os.environ.get("HF_TOKEN")
    files = [f for f in HfApi(token=tok).list_repo_files(args.repo, repo_type="dataset")
             if f.endswith(".parquet") and f.startswith(args.config + "/")]
    print(f"{len(files)} parquet files under {args.config}: {files[:3]}")
    fs = HfFileSystem(token=tok)
    count, hours, cross = Counter(), defaultdict(float), defaultdict(Counter)

    def read(f):
        path = f"datasets/{args.repo}/{f}"
        names = pq.ParquetFile(fs.open(path)).schema_arrow.names
        cols = [c for c in [args.column, "duration", *args.also] if c in names]
        return pq.read_table(fs.open(path), columns=cols).to_pydict()

    from concurrent.futures import ThreadPoolExecutor
    done = 0
    for t in ThreadPoolExecutor(args.threads).map(read, files):
        done += 1
        if done % 25 == 0:
            print(f"  ... {done}/{len(files)} files", flush=True)
        for i, v in enumerate(t[args.column]):
            count[v] += 1
            hours[v] += (t.get("duration") or [0] * len(t[args.column]))[i] / 3600
            for a in args.also:
                if a in t:
                    cross[v][t[a][i]] += 1
    for v, n in count.most_common():
        print(f"  {str(v):10s} {n:>9,} rows {hours[v]:8.1f} h   " +
              ", ".join(f"{k} {m}" for k, m in cross[v].most_common(4)))


if __name__ == "__main__":
    main()
