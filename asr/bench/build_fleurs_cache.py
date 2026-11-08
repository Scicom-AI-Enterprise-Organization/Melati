#!/usr/bin/env python3
"""Materialise the FLEURS sample once, before any checkpoint needs it.

`run_benchmark.py` can build this on first use, but that first use would be a GPU job sitting
idle through a 60-config download -- and in a sweep, six GPU jobs would each start their own.
Build it here, on CPU, while the GPUs train.

**Do not stream this.** `load_dataset(..., streaming=True)` over `google/fleurs` manages about
one language every four minutes: it re-resolves the repo per config and the connection keeps
dropping mid-body. A test shard is only ~200 MB, so fetching the file and reading its first
rows is an order of magnitude faster, and each shard is deleted once its 20 clips are out so
the 60 of them never sit on disk at once.

Rows are re-sorted into the canonical language order, so the cache is identical whatever order
the workers finish in.

    .venv_wild/bin/python bench/build_fleurs_cache.py --workers 8
"""
import argparse, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import FLEURS, FLEURS_PER_LANG  # noqa: E402

REPO = "google/fleurs"


def one_lang(lang, code, per_lang, keep):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    rows = []
    try:
        shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                        if f.startswith(f"parquet-data/{code}/test-") and f.endswith(".parquet"))
        if not shards:
            raise FileNotFoundError(f"no test shard for {code}")
        path = hf_hub_download(REPO, shards[0], repo_type="dataset")
        pf = pq.ParquetFile(path)
        cols = set(pf.schema_arrow.names)
        want = ["audio"] + [c for c in ("transcription", "raw_transcription") if c in cols]
        batch = next(pf.iter_batches(batch_size=per_lang, columns=want)).to_pylist()
        for i, r in enumerate(batch[:per_lang]):
            a = r.get("audio") or {}
            raw = a.get("bytes") if isinstance(a, dict) else None
            if raw is None:
                continue
            rows.append({"id": f"fleurs_{lang}_{i:03d}", "lang": lang, "audio_bytes": raw,
                         "reference_text": r.get("transcription")
                                           or r.get("raw_transcription") or ""})
        del pf
        if not keep:
            # 60 shards x ~200 MB would be 12 GB of cache for 1,200 clips.
            os.unlink(os.path.realpath(path))
        print(f"  [fleurs] {lang} ({code}): {len(rows)}", flush=True)
    except Exception as e:
        print(f"  [fleurs] {lang} ({code}) skipped: {type(e).__name__} {str(e)[:70]}", flush=True)
    return lang, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=FLEURS_PER_LANG)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path("bench/fleurs_sample.parquet"))
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        got = dict(ex.map(lambda kv: one_lang(kv[0], kv[1], a.per_lang, a.keep_shards),
                          FLEURS.items()))

    rows = [r for lang in FLEURS for r in got.get(lang, [])]
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.cache.with_suffix(f".parquet.{os.getpid()}.part")
    pq.write_table(pa.table({
        "id": [r["id"] for r in rows], "lang": [r["lang"] for r in rows],
        "reference_text": [r["reference_text"] for r in rows],
        "audio_bytes": [r["audio_bytes"] for r in rows],
    }), tmp, compression="zstd", use_dictionary=[])
    tmp.rename(a.cache)
    print(f"{len(rows)} clips, {len({r['lang'] for r in rows})} languages, "
          f"{a.cache.stat().st_size/1e6:.0f} MB -> {a.cache}")


if __name__ == "__main__":
    main()
