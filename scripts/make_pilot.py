#!/usr/bin/env python3
"""The pilot: a fixed random subset of recordings both pipelines process, for the comparison.

Writes <out>/sources.jsonl (the Melati stage input) and <out>/emilia_in/<id><ext> symlinks (the
folder upstream Emilia-Pipe walks). Recordings are drawn with a fixed seed, and the draw is
stratified by duration so the long tail (hour-long episodes) is represented, not just the median.

    python scripts/make_pilot.py --sources $MELATI_DATA/podcast/sources.jsonl --n 200 \\
        --out $MELATI_DATA/pilot
"""
import argparse, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.sources.open()]
    rows.sort(key=lambda r: r["bytes"])            # bytes track duration for one codec family
    rng = random.Random(args.seed)
    k = 4                                           # four duration strata, equal draws
    strata = [rows[i * len(rows) // k:(i + 1) * len(rows) // k] for i in range(k)]
    pick = [r for s in strata for r in rng.sample(s, min(len(s), args.n // k))]
    (args.out / "emilia_in").mkdir(parents=True, exist_ok=True)
    with (args.out / "sources.jsonl").open("w") as fh:
        for r in sorted(pick, key=lambda r: r["id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            link = args.out / "emilia_in" / (r["id"] + Path(r["path"]).suffix)
            if not link.exists():
                link.symlink_to(r["path"])
    print(f"{len(pick)} recordings, {sum(r['bytes'] for r in pick) / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
