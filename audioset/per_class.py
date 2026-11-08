#!/usr/bin/env python3
"""One model's average precision on each of AudioSet's 527 classes, on the evaluation set in fp32, with how many of
the evaluation clips carry the class: the per-label table a release card lists (audioset/release.py). The
probabilities are audioset/evaluate.py's, read from its cache when they are there (a model not yet scored is run on
the GPU). The mean over the classes must be the mAP evaluate.py reported. RUNS ON THE BOX, in .venv-sed.

    python audioset/per_class.py --ckpt $MELATI_DATA/models/audioset/frames1/clip.pt \\
        --results $MELATI_DATA/results/audioset/student_v3_fp32.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import ROOT, classes, load  # noqa: E402
from evaluate import run  # noqa: E402

RESULTS = ROOT.parent / "results" / "audioset"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True, help="evaluate.py --fp32's results, which hold its mAP")
    a = ap.parse_args()
    from sklearn.metrics import average_precision_score
    import zoo
    zoo.BF16 = False
    key = f"melati:{a.ckpt}"
    expect = json.loads(a.results.read_text())["models"][key]
    _, _, _, Y = load("eval")
    P = run(key, "eval").astype(np.float32)
    pos = Y.sum(0).astype(int)
    aps = [float(average_precision_score(Y[:, c], P[:, c])) if pos[c] else None for c in range(Y.shape[1])]
    got = float(np.mean([x for x in aps if x is not None]))
    assert abs(got - expect) < 1e-4, (got, expect)
    res = {"model": key, "split": "eval", "clips": int(len(Y)), "fp32": True, "map": round(got, 4),
           "classes": [{"index": i, "mid": mid, "name": n, "ap": None if aps[i] is None else round(aps[i], 4),
                        "positives": int(pos[i])} for i, mid, n in classes()]}
    out = RESULTS / f"per_class_{a.ckpt.parent.name}.json"
    out.write_text(json.dumps(res, indent=1))
    scored = [x for x in aps if x is not None]
    print(f"{key}: mAP {got:.4f} over {len(scored)} classes with a positive clip ({sum(pos == 0)} without); "
          f"AP below 0.2 on {sum(x < 0.2 for x in scored)}, 0.5 or more on {sum(x >= 0.5 for x in scored)} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
