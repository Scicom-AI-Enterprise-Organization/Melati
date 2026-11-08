#!/usr/bin/env python3
"""Where a fine-tune decoded as vLLM serves it goes wrong on speech. RUNS ON THE BOX.

Allowed to stop, a model can stop on speech; allowed nothing else, it can also loop. Per run:

  noise_empty   speech_in_noise clips returned empty, by SNR (dB) and by background
  fleurs_empty  FLEURS clips returned empty, by language
  runaway       reduplication clips where the repeated unit is written over 1.5x as often as it was said
                (score_eval_run.py's rule), with the units it happens on

    python asr/scripts/served_failures.py --runs $MELATI_DATA/asr/runs/base_large_v3 $MELATI_DATA/asr/runs2/v3_* \\
        --out $MELATI_DATA/results/asr/served_failures.json
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from metrics import normalise, overgeneration_ratio  # noqa: E402


def rows(run: Path, arm: str):
    f = next(iter(sorted((run / "bench").glob(f"*/{arm}.jsonl"))), None)
    return [json.loads(l) for l in f.open(encoding="utf-8")] if f else []


def current(rs):
    return bool(rs) and all(r.get("suppress") == "none" for r in rs[:1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for run in args.runs:
        res = {}
        sin = rows(run, "speech_in_noise")
        if current(sin):
            by_snr, by_bg = defaultdict(list), defaultdict(list)
            for r in sin:
                m = r.get("meta") or {}
                e = not normalise(r["hyp"])
                by_snr[m.get("snr_db")].append(e)
                by_bg[m.get("background")].append(e)
            res["noise_empty_by_snr"] = {str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_snr.items())}
            res["noise_empty_by_background"] = {str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_bg.items())}
            res["noise_n_by_snr"] = {str(k): len(v) for k, v in sorted(by_snr.items())}
        fl = rows(run, "fleurs")
        if current(fl):
            by = defaultdict(list)
            for r in fl:
                by[(r.get("meta") or {}).get("lang")].append(not normalise(r["hyp"]))
            res["fleurs_empty_by_lang"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())}
        rd = rows(run, "reduplication")
        if current(rd):
            run_away = []
            for r in rd:
                m = r.get("meta") or {}
                if overgeneration_ratio(r["hyp"], m.get("unit", ""), int(m.get("n_repeats", 0) or 0)) > 1.5:
                    run_away.append(m.get("unit", ""))     # score_eval_run.py's rule, the repeated unit
            res["runaway"] = {"n": len(rd), "count": len(run_away), "rate": round(len(run_away) / len(rd), 4),
                              "top_repeated": Counter(run_away).most_common(8)}
        if res:
            out[f"{run.parent.name}/{run.name}"] = res
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out.items():
        print(k)
        if "noise_empty_by_snr" in v:
            print("   noise empty by SNR:", v["noise_empty_by_snr"], " by bg:", v["noise_empty_by_background"])
        if "runaway" in v:
            print("   runaway:", v["runaway"]["count"], "/", v["runaway"]["n"], v["runaway"]["top_repeated"][:5])
        if "fleurs_empty_by_lang" in v:
            worst = sorted(v["fleurs_empty_by_lang"].items(), key=lambda kv: -kv[1])[:8]
            print("   fleurs empty, worst languages:", worst)


if __name__ == "__main__":
    main()
