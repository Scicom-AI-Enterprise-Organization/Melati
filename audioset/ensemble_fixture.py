#!/usr/bin/env python3
"""A small fixture for checking Melati-AudioSet-Ensemble outside .venv-sed: N clips of AudioSet's evaluation set and
the fp32 probabilities audioset/evaluate.py cached for each member on them, in one .npz that numpy alone reads
(audioset/check_ensemble_fresh.py). RUNS ON THE BOX, in .venv-sed.

    python audioset/ensemble_fixture.py --n 64 --out $MELATI_DATA/tmp/ensemble_fixture.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from data import load  # noqa: E402
from evaluate import PROBS  # noqa: E402
from release_ensemble import MEMBERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W, ids, lens, Y = load("eval")
    pick = np.linspace(0, len(W) - 1, a.n).round().astype(int)          # spread over the set, not its first clips
    out = {"wave": np.stack([np.asarray(W[i], np.float32) for i in pick]), "index": pick, "ids": ids[pick],
           "lens": lens[pick], "y": Y[pick]}
    for m in MEMBERS:
        p = np.load(PROBS / f"eval_{m['zoo'].replace('melati:', 'melati_').replace('/', '_')}.fp32.npy")
        out[f"p_{m['name']}"] = p[pick].astype(np.float32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"{a.out}: {a.n} clips, members {[m['name'] for m in MEMBERS]}", flush=True)


if __name__ == "__main__":
    main()
