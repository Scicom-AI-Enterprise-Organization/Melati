#!/usr/bin/env python3
"""A development set the pipeline can be run end to end on while the podcasts download.

VoxConverse (diarizers-community/voxconverse, CC BY 4.0): YouTube debates, news and talk shows
with music, effects and crowd noise, and reference speaker turns. Writes N recordings as WAV plus
a sources.jsonl in Melati's format, and the reference turns as JSON (start, end, speaker) for the
diarization and purity checks.

    python scripts/dev_voxconverse.py --parquet-dir $MELATI_DATA/bench/voxconverse/data \\
        --split test --n 12 --out $MELATI_DATA/dev/vox
"""
import argparse, io, json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(args.parquet_dir.glob(f"{args.split}-*.parquet")):
        t = pq.ParquetFile(f)
        for b in t.iter_batches(batch_size=4):
            for r in b.to_pylist():
                if len(rows) >= args.n:
                    break
                audio = r["audio"]
                x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                rid = f"vox{len(rows):03d}"
                path = args.out / f"{rid}.wav"
                sf.write(path, x.mean(1), sr)
                turns = [[round(float(s), 3), round(float(e), 3), str(k)] for s, e, k in
                         zip(r["timestamps_start"], r["timestamps_end"], r["speakers"])]
                (args.out / f"{rid}.ref.json").write_text(json.dumps({"turns": turns}))
                rows.append({"id": rid, "path": str(path), "stem": rid, "bytes": path.stat().st_size,
                             "duration_s": round(len(x) / sr, 2)})
    with (args.out / "sources.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} recordings, {sum(r['duration_s'] for r in rows) / 3600:.2f} h -> {args.out}")


if __name__ == "__main__":
    main()
