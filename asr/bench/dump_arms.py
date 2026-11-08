#!/usr/bin/env python3
"""Write benchmark arms to plain numpy files, for vllm_check.py. RUNS ON THE BOX (.venv-asr).

The vLLM environments carry no `datasets`, and the arms are read exactly as run_benchmark.py reads them
(same loaders, same FLEURS sample), so the two decoders see the same clips in the same order.

    python asr/bench/dump_arms.py --arms silence music nonspeech fleurs --out $MELATI_DATA/asr/arms_npz \\
        --fleurs-cache $MELATI_DATA/asr/bench/fleurs_sample.parquet
"""
import argparse, io, json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from run_benchmark import FLEURS_PER_LANG, REPO, SR, load_cv22_test, load_fleurs, reference_of


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fleurs-cache", type=Path, default=Path("bench/fleurs_sample.parquet"))
    ap.add_argument("--cv22-test", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        if arm == "fleurs":
            ds = load_fleurs(FLEURS_PER_LANG, 0, args.fleurs_cache)
        elif arm == "cv22":
            ds = load_cv22_test(args.cv22_test, 0)
        else:
            ds = load_dataset(REPO, arm, split="test").cast_column("audio", Audio(decode=False))
        waves, meta = [], []
        for i in range(len(ds)):
            r = ds[i]
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            assert sr == SR, f"{r.get('id')} is {sr} Hz"
            waves.append(x if x.ndim == 1 else x.mean(1))
            meta.append({"id": r.get("id", ""), "lang": r.get("lang") or (r.get("meta") or {}).get("lang"),
                         "reference_text": reference_of(r, arm)})
        offsets = np.cumsum([0] + [len(w) for w in waves])
        np.savez(args.out / f"{arm}.npz", audio=np.concatenate(waves).astype(np.float32), offsets=offsets)
        (args.out / f"{arm}.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta),
                                               encoding="utf-8")
        print(f"{arm}: {len(waves)} clips, {offsets[-1] / SR / 3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
