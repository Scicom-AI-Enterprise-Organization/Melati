#!/usr/bin/env python3
"""Average the held-out probabilities of several emotion models and score the mix. RUNS ON THE BOX.

Inputs are .npz files with ids, labels (y) and probabilities: emotion/probe.py --save-probs, and
emotion/score_current.py's (one array per current model, chosen with name=key). Clips are matched by id, so
every member is scored on the same clips; the scores and thresholds are probe.py's.

    python emotion/ensemble.py --members $MELATI_DATA/results/emotion/probs_w2vbert.npz \\
        $MELATI_DATA/results/emotion/current.npz:meralion+e2v --out $MELATI_DATA/results/emotion/ensemble.json
"""
import argparse, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import LABELS, PRIORS, scores, thresholds  # noqa: E402


def load(spec):
    path, _, key = spec.partition(":")
    z = np.load(path, allow_pickle=False)
    p = z[key] if key else z["p"]
    name = key or Path(path).stem.replace("probs_", "")
    return name, dict(zip(z["ids"].tolist(), p)), dict(zip(z["ids"].tolist(), z["y"].tolist()))


def thresholds_cv(P, y, ids, folds=2):
    """Thresholds chosen on one half of the clips and measured on the other (then the reverse), averaged: choosing
    and measuring on the same clips flatters recall. Halves are fixed by clip id, so every model gets the same."""
    import hashlib
    f = np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % folds for i in ids])
    out = {}
    for k in range(folds):
        tr, te = f != k, f == k
        t = thresholds(P[tr], y[tr])
        for lab, v in t.items():
            if v is None:
                out.setdefault(lab, []).append(None)
                continue
            c = LABELS.index(lab)
            pos, neg = P[te & (y == c), c], P[te & (y != c), c]
            tp, fp = (pos >= v["threshold"]).mean(), (neg >= v["threshold"]).mean()
            prior = PRIORS[lab]
            out.setdefault(lab, []).append({"recall": float(tp),
                                            "precision_at_prior": float(prior * tp / (prior * tp + (1 - prior) * fp + 1e-12))})
    res = {}
    for lab, vs in out.items():
        if any(v is None for v in vs):
            res[lab] = None
        else:
            res[lab] = {k: round(float(np.mean([v[k] for v in vs])), 4) for k in ("recall", "precision_at_prior")}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    mem = [load(m) for m in args.members]
    ids = sorted(set.intersection(*[set(p) for _, p, _ in mem]))
    y = np.array([mem[0][2][i] for i in ids])
    res = {}
    for k in range(1, len(mem) + 1):
        for combo in combinations(mem, k):
            name = " + ".join(n for n, _, _ in combo)
            P = np.mean([np.stack([p[i] for i in ids]) for _, p, _ in combo], 0)
            res[name] = {"held_out": scores(P, y), "thresholds": thresholds(P, y),
                         "thresholds_cv": thresholds_cv(P, y, ids)}
            t = {c: v and v["recall"] for c, v in res[name]["thresholds"].items()}
            print(f"{name:45s} UAR {res[name]['held_out']['uar']:.4f} acc {res[name]['held_out']['acc']:.4f}  recall at "
                  f"0.75 precision: {t}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n": len(ids), "results": res}, indent=1))


if __name__ == "__main__":
    main()
