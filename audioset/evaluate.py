#!/usr/bin/env python3
"""AudioSet taggers on the same clips: each model's probabilities on a cached split (audioset/data.py), its mAP (the
mean over classes with a positive of their average precision, as AudioSet results are reported), and the mAP of every
average of the models' probabilities, the candidate teachers to distil from. RUNS ON THE BOX, in .venv-sed.

    python audioset/evaluate.py --models ced-base eat-large eat-base sslam ast --split eval \\
        --out $MELATI_DATA/results/audioset/teachers_eval.json
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, load  # noqa: E402

PROBS = ROOT / "probs"


def mean_ap(P, Y):
    """Over the classes with a positive that the probabilities cover (a PretrainedSED tagger has 447 of the 527)."""
    from sklearn.metrics import average_precision_score
    keep = (Y.sum(0) > 0) & ~np.isnan(P).any(0)
    return float(np.mean([average_precision_score(Y[:, c], P[:, c]) for c in np.flatnonzero(keep)]))


def run(name, split, batch=64):
    import zoo
    tag = name.replace("melati:", "melati_").replace("/", "_")             # a checkpoint's path as a file name
    out = PROBS / f"{split}_{tag}{'' if zoo.BF16 else '.fp32'}.npy"
    if out.exists():
        return np.load(out)
    from zoo import load as load_model
    W, _, _, _ = load(split)
    m = load_model(name)
    t0, P = time.time(), []
    for i in range(0, len(W), batch):
        P.append(m([w.astype(np.float32) for w in W[i:i + batch]]).astype(np.float16))
    P = np.concatenate(P)
    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(out, P)
    print(f"{name}: {len(P)} clips in {time.time() - t0:.0f} s", flush=True)
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast (probabilities cached under <model>.fp32)")
    ap.add_argument("--add-to", default=None, metavar="A,B,...",
                    help="average these models with every subset of the others, instead of every combination of all")
    ap.add_argument("--averages", nargs="+", default=None, metavar="A,B,...",
                    help="average only these combinations, each its models joined by commas")
    a = ap.parse_args()
    if a.fp32:
        import zoo
        zoo.BF16 = False
    _, ids, _, Y = load(a.split)
    probs = {}
    for name in a.models:
        try:
            probs[name] = run(name, a.split).astype(np.float32)
        except Exception as e:                                       # a model that will not load is reported, not fatal
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    res = {"split": a.split, "clips": int(len(ids)), "classes_scored": int((Y.sum(0) > 0).sum()),
           "models": {n: round(mean_ap(P, Y), 4) for n, P in probs.items()}, "averages": {}}
    partial = {n: int(((Y.sum(0) > 0) & ~np.isnan(P).any(0)).sum()) for n, P in probs.items() if np.isnan(P).any()}
    if partial:
        res["classes_scored_by"] = partial                  # a model scored on fewer classes than the rest
    for n, v in sorted(res["models"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:10s} mAP {v:.4f}", flush=True)
    if a.averages:
        combos = [tuple(sorted(c.split(","))) for c in a.averages]
        missing = sorted({n for c in combos for n in c} - set(probs))
        assert not missing, f"--averages names models not scored: {missing}"
    elif a.add_to:
        base = a.add_to.split(",")
        extra = sorted(n for n in probs if n not in base)
        combos = [tuple(sorted(base + list(x))) for k in range(len(extra) + 1) for x in itertools.combinations(extra, k)]
    else:
        combos = [c for k in range(2, len(probs) + 1) for c in itertools.combinations(sorted(probs), k)]
    for combo in combos:                            # each model averaged only where it has the class
        res["averages"][" + ".join(combo)] = round(mean_ap(np.nanmean([probs[c] for c in combo], 0), Y), 4)
    best = sorted(res["averages"].items(), key=lambda kv: -kv[1])[:8]
    for n, v in best:
        print(f"  {v:.4f}  {n}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
