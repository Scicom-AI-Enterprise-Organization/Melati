#!/usr/bin/env python3
"""Which decoding each benchmark arm of a run was scored with. RUNS ON THE BOX.

run_benchmark.py decodes as vLLM does (no suppressed tokens) unless given --hf-suppress, and writes the
mode into every row as `suppress`; rows written before that field existed are `unknown` (Hugging Face's
default, which forbids <|endoftext|> as the first token). A run is `current` when every arm it should
have carries `none`.

    python asr/scripts/bench_protocol.py $MELATI_DATA/asr/runs2/v3_* [--stale-only]
"""
import argparse, json
from pathlib import Path

ARMS = ("silence", "music", "nonspeech", "reduplication", "speech_in_noise", "genuine", "genuine_isolated",
        "librispeech_test_clean", "fleurs", "cv22", "lexicon_synth", "wild", "fleurs_forced")


def modes(run: Path) -> dict:
    out = {}
    for arm in ARMS:
        fs = sorted((run / "bench").glob(f"*/{arm}.jsonl"))
        if not fs:
            out[arm] = "missing"
            continue
        with fs[0].open(encoding="utf-8") as fh:
            first = fh.readline()
        out[arm] = json.loads(first).get("suppress", "unknown") if first.strip() else "empty"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--stale-only", action="store_true", help="print only the runs not yet scored as vLLM decodes")
    args = ap.parse_args()
    for run in args.runs:
        m = modes(run)
        current = all(v == "none" for v in m.values())
        if args.stale_only:
            if not current:
                print(run)
            continue
        odd = {k: v for k, v in m.items() if v != "none"}
        print(f"{'current' if current else 'stale  '}  {run.name}  {odd if odd else ''}")


if __name__ == "__main__":
    main()
