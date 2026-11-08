#!/usr/bin/env python3
"""Why is a fine-tune worse than base on an arm? Clip by clip, against base on the same clips.

A multilingual regression has two very different causes: the model still transcribes but less well
(CER up a little on many clips), or it answers in the WRONG language or translates (CER near 1 on
a few clips, usually English or Malay where the clip is not). The first is forgetting; the second
is language identification drifting towards the fine-tune's dominant languages, and is fixed by
other means (forcing the language, or language-balanced data). This counts both, per language,
and prints the worst clips side by side.

    python asr/diagnose_regression.py --runs $MELATI_DATA/asr/runs --run <name> --arm fleurs
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
from metrics import cer, normalise  # noqa: E402


def load(runs: Path, run: str, arm: str):
    f = next((runs / run / "bench").rglob(f"{arm}.jsonl"))
    return {r["audio_filepath"]: r for r in map(json.loads, f.open(encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="base_large_v3")
    ap.add_argument("--arm", default="fleurs")
    ap.add_argument("--blowup", type=float, default=0.8, help="a clip at or above this CER failed outright")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    b, r = load(args.runs, args.base, args.arm), load(args.runs, args.run, args.arm)
    per = defaultdict(lambda: {"n": 0, "base_cer": 0.0, "run_cer": 0.0, "blowup_base": 0, "blowup_run": 0})
    worst = []
    for k in sorted(set(b) & set(r)):
        ref = normalise(b[k].get("reference_text") or "")
        if not ref:
            continue
        lang = (b[k].get("meta") or {}).get("lang", "?")
        cb, cr = cer(ref, normalise(b[k]["hyp"])), cer(ref, normalise(r[k]["hyp"]))
        p = per[lang]
        p["n"] += 1; p["base_cer"] += cb; p["run_cer"] += cr
        p["blowup_base"] += cb >= args.blowup; p["blowup_run"] += cr >= args.blowup
        worst.append((cr - cb, lang, k, b[k]["hyp"], r[k]["hyp"]))
    tot = {key: sum(p[key] for p in per.values()) for key in ("n", "blowup_base", "blowup_run")}
    macro = lambda k: sum(p[k] / p["n"] for p in per.values()) / max(len(per), 1)
    print(f"{args.arm}: {tot['n']} clips, {len(per)} languages; failed outright (CER >= {args.blowup}): "
          f"base {tot['blowup_base']}, {args.run} {tot['blowup_run']}; macro CER (mean of clip CERs per "
          f"language) base {macro('base_cer'):.4f} -> run {macro('run_cer'):.4f}")
    print(f"{'lang':6} {'n':>3} {'base':>7} {'run':>7} {'delta':>7}  blowups base->run")
    for lang, p in sorted(per.items(), key=lambda kv: -(kv[1]["run_cer"] - kv[1]["base_cer"]) / kv[1]["n"])[:20]:
        n = p["n"]
        print(f"{lang:6} {n:3d} {p['base_cer'] / n:7.3f} {p['run_cer'] / n:7.3f} {(p['run_cer'] - p['base_cer']) / n:+7.3f}"
              f"  {p['blowup_base']} -> {p['blowup_run']}")
    print("\nworst clips (run minus base CER):")
    for d, lang, k, hb, hr in sorted(worst, reverse=True)[: args.examples]:
        print(f"  [{lang}] {d:+.2f}\n     base: {hb[:110]}\n     run:  {hr[:110]}")


if __name__ == "__main__":
    main()
