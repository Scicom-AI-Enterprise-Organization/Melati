#!/usr/bin/env python3
"""Two seven-model emotion averages side by side, per held-out corpus. RUNS ON THE BOX.

emotion/ensemble.py scores a mix on all held-out clips at once; nEMO (acted Polish) and the VoxCeleb half of
NonverbalTTS (English interviews, the nearer to podcast speech) can disagree, so they are scored apart here.

    python scripts/compare_emotion_ensembles.py --a <npz members...> --b <npz members...> --out x.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "emotion"))
from ensemble import load  # noqa: E402
from probe import scores  # noqa: E402


def mix(specs):
    ms = [load(s) for s in specs]
    ids = sorted(set.intersection(*[set(m[1]) for m in ms]))
    P = np.mean([[m[1][i] for i in ids] for m in ms], 0)
    return ids, P, np.array([ms[0][2][i] for i in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--names", nargs=2, default=["a", "b"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {}
    for name, specs in zip(args.names, (args.a, args.b)):
        ids, P, y = mix(specs)
        corpus = np.array([i.split("/")[0] for i in ids])
        res[name] = {"all": scores(P, y)}
        for c in sorted(set(corpus)):
            m = corpus == c
            res[name][c] = scores(P[m], y[m])
        print(name, {c: round(r["uar"], 4) for c, r in res[name].items()}, flush=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
