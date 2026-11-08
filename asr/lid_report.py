#!/usr/bin/env python3
"""FLEURS with the language detected and with it forced, for every scored run: what language
identification costs a fine-tune, apart from what it costs to transcribe. RUNS ON THE BOX.

Per run: macro CER (the mean over languages of the mean clip CER), clips that fail outright
(CER >= 0.8), and per-language CER, for `fleurs.jsonl` (the model picks the language) and
`fleurs_forced.jsonl` (each clip decoded with its own language token); and, where the bench recorded
the token it decoded with (`lid`), language-identification accuracy and what the misses became.

    python asr/lid_report.py --roots $MELATI_DATA/asr/runs $MELATI_DATA/asr/runs2 \\
        --out $MELATI_DATA/results/asr/lid.json
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
from metrics import cer, normalise  # noqa: E402


def score(path: Path, blowup=0.8):
    per = defaultdict(list)
    lid_ok, lid_n, wrong = 0, 0, Counter()
    for r in map(json.loads, path.open(encoding="utf-8")):
        ref = normalise(r.get("reference_text") or "")
        if not ref:
            continue
        lang = (r.get("meta") or {}).get("lang", "?")
        per[lang].append(cer(ref, normalise(r["hyp"])))
        if r.get("lid") and not r.get("forced_language"):
            lid_n += 1
            lid_ok += r["lid"] == lang
            if r["lid"] != lang:
                wrong[r["lid"]] += 1
    langs = {k: round(sum(v) / len(v), 4) for k, v in sorted(per.items())}
    out = {"n": sum(len(v) for v in per.values()), "n_langs": len(per),
           "macro_cer": round(sum(langs.values()) / max(len(langs), 1), 4),
           "failed": sum(c >= blowup for v in per.values() for c in v), "per_lang": langs}
    if lid_n:
        out["lid_accuracy"] = round(lid_ok / lid_n, 4)
        out["lid_wrong_as"] = dict(wrong.most_common(8))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {}
    for root in args.roots:
        for run in sorted(p for p in root.iterdir() if (p / "bench").is_dir()):
            arms = {}
            for arm in ("fleurs", "fleurs_forced"):
                f = next(iter(sorted((run / "bench").rglob(f"{arm}.jsonl"))), None)
                if f:
                    arms[arm] = score(f)
            if arms:
                meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
                res[f"{root.name}/{run.name}"] = {"prefix": meta.get("prefix", "masked (bug)" if meta else None),
                                                  "lid_loss": meta.get("lid_loss"), **arms}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        a, f = v.get("fleurs") or {}, v.get("fleurs_forced") or {}
        print(f"{k:62s} detected {a.get('macro_cer')} ({a.get('failed')} failed)  "
              f"forced {f.get('macro_cer')} ({f.get('failed')} failed)  lid_acc {a.get('lid_accuracy')}")


if __name__ == "__main__":
    main()
