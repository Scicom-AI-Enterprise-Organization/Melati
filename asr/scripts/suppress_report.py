#!/usr/bin/env python3
"""What token suppression does to the voice-free arms: Hugging Face's default against vLLM's. RUNS ON THE BOX.

Reads deploy/box_asr_suppress_check.sh's output (<root>/<model>/<vllm|hf>/<tag>/<arm>.jsonl) and writes,
per model, mode and arm, the share of clips with any word written, the share with any output at all, and
the commonest transcripts, so the mechanism is visible rather than inferred.

    python asr/scripts/suppress_report.py --root $MELATI_DATA/asr/suppress_check \\
        --out $MELATI_DATA/results/asr/suppress_check.json
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from metrics import normalise  # noqa: E402

ARMS = ("silence", "music", "nonspeech")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for mdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for mode in ("hf", "vllm"):
            for arm in ARMS:
                fs = sorted((mdir / mode).glob(f"*/{arm}.jsonl"))
                if not fs:
                    continue
                rows = [json.loads(l) for l in fs[0].open(encoding="utf-8")]
                hyps = [r["hyp"].strip() for r in rows]
                words = [bool(normalise(h)) for h in hyps]
                out.setdefault(mdir.name, {}).setdefault(mode, {})[arm] = {
                    "n": len(rows),
                    "words_rate": round(sum(words) / len(rows), 4),
                    "any_output_rate": round(sum(bool(h) for h in hyps) / len(rows), 4),
                    "top": Counter(hyps).most_common(6),
                }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for m, modes in out.items():
        for mode, arms in modes.items():
            cells = "  ".join(f"{a} {v['words_rate']:.3f}/{v['any_output_rate']:.3f}" for a, v in arms.items())
            print(f"{m:10s} {mode:5s} words/any  {cells}")
            for a, v in arms.items():
                print(f"{'':17s}{a:10s} {v['top'][:4]}")


if __name__ == "__main__":
    main()
