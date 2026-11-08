#!/usr/bin/env python3
"""The served heads reproduce the measured ones. RUNS ON THE BOX (.venv-melati, the tag server's environment).

Loads each head the way the tag server does (melati.emotion.load("h_<encoder>")), runs it on held-out clips decoded
the way the features were (emotion/extract.py's Audio), and compares with the probabilities emotion/probe.py saved
for the same clips. Agreement should be near-total; a gap means serving and training compute different features.

    python emotion/check_serving.py --manifest $MELATI_DATA/emo/manifest.jsonl \\
        --probs-dir $MELATI_DATA/results/emotion --n 64
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract import Audio  # noqa: E402
from melati import emotion as EMO  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--probs-dir", type=Path, required=True)
    ap.add_argument("--encoders", nargs="+", default=["whisper", "whisper_ft", "w2vbert", "wavlm", "e2v"])
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    rows = {r["id"]: r for r in map(json.loads, open(args.manifest, encoding="utf-8")) if r["split"] == "test"}
    audio = Audio()
    for enc in args.encoders:
        z = np.load(args.probs_dir / f"probs_{enc}.npz")
        ids, P = z["ids"][: args.n], z["p"][: args.n]
        model = EMO.load(f"h_{enc}")
        clips = [audio(rows[i]["src"]) for i in ids.tolist()]
        Q = np.concatenate([model(clips[s:s + 16])[0] for s in range(0, len(clips), 16)])
        same = float((P.argmax(1) == Q.argmax(1)).mean())
        print(f"h_{enc:10s} same top class {same:.3f}  max |dp| {float(np.abs(P - Q).max()):.4f}  "
              f"mean |dp| {float(np.abs(P - Q).mean()):.5f}", flush=True)
        del model


if __name__ == "__main__":
    main()
