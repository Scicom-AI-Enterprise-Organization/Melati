#!/usr/bin/env python3
"""Check Melati's dataset the way a user would load it: `datasets` sees `audio` as 24 kHz Audio and every column with
one type across all shards, and sampled rows decode to the length their row states with every tag of the
annotation in the tagged text. --data reads --rows rows from EVERY local shard; --repo streams the published one.
RUNS ON THE BOX.

    python scripts/check_hf_dataset.py --data $MELATI_DATA/podcast/hf/data
    python scripts/check_hf_dataset.py --repo Scicom-intl/Melati
"""
import argparse
import io
import json
from pathlib import Path

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=None)
ap.add_argument("--repo", default=None)
ap.add_argument("--rows", type=int, default=20, help="per shard with --data, in all with --repo")
args = ap.parse_args()
from datasets import Audio, load_dataset  # noqa: E402


def ok(r):
    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    ann = json.loads(r["annotation"])
    good = sr == 24000 and abs(len(x) / sr - r["duration"]) < 0.05 and isinstance(ann.get("words"), list)
    good = good and all(f"[{e['kind']}]" in r["tagged_text"] for e in ann.get("events") or [])
    if not good:
        print("BAD", r["id"], sr, round(len(x) / sr, 3), r["duration"], (r["tagged_text"] or "")[:120])
    return good


if args.repo:
    ds = load_dataset(args.repo, split="train", streaming=True)
    sample = list(ds.cast_column("audio", Audio(decode=False)).take(args.rows))
    shards = "the repo"
else:
    import pyarrow.parquet as pq
    files = sorted(args.data.glob("*.parquet"))
    ds = load_dataset("parquet", data_files=[str(f) for f in files], split="train", streaming=True)
    schemas = {str(pq.read_schema(f).remove_metadata()) for f in files}
    assert len(schemas) == 1, f"{len(schemas)} different schemas across {len(files)} shards"
    sample = [r for f in files for r in pq.ParquetFile(f).iter_batches(batch_size=args.rows).__next__().to_pylist()]
    shards = f"{len(files)} shards"
assert isinstance(ds.features["audio"], Audio) and ds.features["audio"].sampling_rate == 24000, ds.features["audio"]
bad = sum(not ok(r) for r in sample)
r = sample[-1]
print(f"{len(sample)} rows from {shards}, {bad} bad; e.g. {r['id']} {r['language']} {r['duration']}s: "
      f"{(r['tagged_text'] or '')[:160]}")
print("CHECK_OK" if bad == 0 and sample else "CHECK_FAILED", flush=True)
import os  # noqa: E402
os._exit(0 if bad == 0 and sample else 1)     # streaming's background threads can abort the interpreter's shutdown
