#!/usr/bin/env python3
"""Speech clips a checkpoint returned NOTHING for, and the error rate without them. RUNS ON THE BOX.

A model allowed to stop (decoded as vLLM serves it) can stop on speech too. This counts, per speech arm,
the clips with a reference and an empty transcript, what share of the arm's error they make up, and the
worst non-empty clips, so a regression can be told apart: stopping early, or transcribing worse.

    python asr/scripts/empty_on_speech.py $MELATI_DATA/asr/runs2/<run> [--arms librispeech_test_clean fleurs]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from metrics import cer, normalise, wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--arms", nargs="+", default=["librispeech_test_clean", "fleurs", "cv22", "genuine"])
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    for run in args.runs:
        for arm in args.arms:
            fs = sorted((run / "bench").glob(f"*/{arm}.jsonl"))
            if not fs:
                continue
            rows = [r for r in map(json.loads, fs[0].open(encoding="utf-8")) if normalise(r.get("reference_text") or "")]
            score = wer if arm in ("librispeech_test_clean", "genuine") else cer
            errs = [(score(r["reference_text"], r["hyp"]), r) for r in rows]
            empty = [e for e in errs if not normalise(e[1]["hyp"])]
            full = sum(e for e, _ in errs) / len(errs)
            rest = [e for e, _ in errs if (e, _) not in empty] if False else [e for e, r in errs if normalise(r["hyp"])]
            print(f"{run.name[:58]:58s} {arm:24s} n={len(rows):5d} empty={len(empty):4d} "
                  f"({100 * len(empty) / len(rows):.2f}%)  mean err {full:.4f}, without empties "
                  f"{sum(rest) / max(1, len(rest)):.4f}")
            for e, r in sorted(errs, key=lambda x: -x[0])[:args.show]:
                print(f"      {e:.2f}  ref: {r['reference_text'][:70]!r}\n            hyp: {r['hyp'][:70]!r}")


if __name__ == "__main__":
    main()
