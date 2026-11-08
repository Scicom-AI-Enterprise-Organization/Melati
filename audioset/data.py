#!/usr/bin/env python3
"""AudioSet as agkphysics/AudioSet ships it (deploy/box_audioset_fetch.sh: parquet shards of 10 s FLAC clips at 44.1
or 48 kHz, stereo, each with its labels as AudioSet ids), or its strongly labelled subset as CLAPv2/audioset_strong
does (deploy/box_audioset_strong_fetch.sh: 48 kHz, every event timed). A split is decoded once to mono 16 kHz and
cached as one float16 array of 10 s rows (a shorter clip padded with zeros, its length kept), with the labels as a
527-column matrix in AudioSet's own class order (class_labels_indices.csv), so every model is scored on the same
samples. RUNS ON THE BOX.

    python audioset/data.py cache --split eval
    python audioset/data.py cache --split strong_test --src $MELATI_DATA/audioset/strong/data/test
"""
import os
import argparse
import csv
import io
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

ROOT = Path(f"{MELATI_DATA}/audioset")
SR, CLIP = 16000, 160000


def classes():
    """[(index, mid, display name)] in AudioSet's order."""
    with open(ROOT / "class_labels_indices.csv", newline="") as f:
        return [(int(r["index"]), r["mid"], r["display_name"]) for r in csv.DictReader(f)]


def decode(b):
    import soundfile as sf
    import soxr
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    x = x.mean(1)
    return soxr.resample(x, sr, SR) if sr != SR else x


MID = re.compile(r"\((/[mtg]/[^)]+)\)")
EVENT = re.compile(r"\((\d+\.\d+)s - (\d+\.\d+)s\) (.+?) \((/[mtg]/[^)]+)\)")


def rows(path):
    """A shard's clips as (YouTube id, audio bytes, clip labels as AudioSet ids, timed events or None), from either
    mirror: agkphysics/AudioSet (video_id, audio, labels) or CLAPv2/audioset_strong (index ending in the id, audio,
    raw_text: 'Class labels: name (mid), ...' and 'Strong events: (0.000s - 3.654s) name (mid) | ...')."""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    if "video_id" in names:
        for r in pq.read_table(path, columns=["video_id", "audio", "labels"]).to_pylist():
            yield r["video_id"], r["audio"]["bytes"], r["labels"], None
        return
    for r in pq.read_table(path, columns=["index", "audio", "raw_text"]).to_pylist():
        rt = {t.split(":", 1)[0]: t.split(":", 1)[1] for t in r["raw_text"] if ":" in t}
        strong = [(float(a), float(b), mid) for a, b, _, mid in EVENT.findall(rt.get("Strong events", ""))]
        yield r["index"][-11:], r["audio"]["bytes"], MID.findall(rt.get("Class labels", "")), strong


def _shard(path):
    ids, wavs, lens, labs, strong, n = [], [], [], [], [], 0
    for vid, b, lab, st in rows(path):
        n += 1
        try:
            x = decode(b)[:CLIP]
        except Exception:                                   # a clip that does not decode is left out, and counted
            continue
        ids.append(vid); lens.append(len(x)); labs.append(lab); strong.append(st)
        wavs.append(np.pad(x, (0, CLIP - len(x))).astype(np.float16))
    W = np.stack(wavs) if wavs else np.zeros((0, CLIP), np.float16)
    return ids, W, lens, labs, n, (strong if any(st is not None for st in strong) else None)


def cache(split, workers=32, src=None):
    """Decode a split once: agk's by name, or any parquet folder (src) under that name, e.g. the strong subset's
    test clips as strong_test, whose timed events are kept beside the labels."""
    files = sorted((Path(src) if src else ROOT / "agk" / "data" / split).glob("*.parquet"))
    mid = {m: i for i, m, _ in classes()}
    out = ROOT / "cache"
    out.mkdir(parents=True, exist_ok=True)
    ids, lens, labs, n_rows, parts, strong = [], [], [], 0, [], []
    with ProcessPoolExecutor(workers) as ex:
        for i, (a, w, l, y, n, st) in enumerate(ex.map(_shard, files), 1):
            ids += a; lens += l; labs += y; n_rows += n; parts.append(w)
            strong += [json.dumps(e) for e in st] if st is not None else ["null"] * len(a)
            print(f"{split}: {i}/{len(files)} shards, {len(ids)} clips", flush=True)
    W = np.concatenate(parts)
    Y = np.zeros((len(ids), len(mid)), np.bool_)
    for k, y in enumerate(labs):
        for m in y:
            if m in mid:
                Y[k, mid[m]] = True
    np.save(out / f"{split}_wav16k.npy", W)
    np.savez(out / f"{split}_meta.npz", ids=np.array(ids), lens=np.array(lens), labels=Y, strong=np.array(strong))
    meta = {"split": split, "rows": n_rows, "clips": len(ids), "undecodable": n_rows - len(ids),
            "hours": round(sum(lens) / SR / 3600, 2), "classes_with_positives": int((Y.sum(0) > 0).sum())}
    (out / f"{split}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def load(split):
    """-> (16 kHz float16 rows, ids, lengths, 527-column labels), memory-mapped."""
    m = np.load(ROOT / "cache" / f"{split}_meta.npz")
    return np.load(ROOT / "cache" / f"{split}_wav16k.npy", mmap_mode="r"), m["ids"], m["lens"], m["labels"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["cache"])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--src", default=None, help="a parquet folder other than agk's split, e.g. the strong test clips")
    a = ap.parse_args()
    cache(a.split, a.workers, a.src)


if __name__ == "__main__":
    main()
