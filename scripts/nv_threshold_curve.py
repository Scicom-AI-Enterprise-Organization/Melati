#!/usr/bin/env python3
"""Tag precision and recall against threshold, per class, on both splits of a saved
scripts/eval_nonverbal_gaps.py run: what each class can reach, not only what clears one target.

    python scripts/nv_threshold_curve.py results/nonverbal/nonverbaltts_gaps.json [--classes breath laugh cough]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_nonverbal_gaps import TO_CLASS, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--classes", nargs="+", default=["breath", "laugh", "cough"])
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="also write the curves as JSON")
    args = ap.parse_args()
    curves = {}
    d = json.loads(args.path.read_text())
    # rebuild the clip lists from the saved candidates (clips without a candidate change no score)
    splits = {"train": [], "test": []}
    for c in d["candidates"]:
        splits[c["split"]].append({"aligned": True, "tags": [tuple(t) for t in c["ref_tags"]], "cands": c["cands"]})
    classes = sorted(set(TO_CLASS.values()))
    for cls in args.classes:
        print(f"== {cls}")
        for t in np.arange(0.05, 0.96, 0.05):          # the grid eval_nonverbal_gaps.calibrate uses
            thr = {c: (round(float(t), 2) if c == cls else 2.0) for c in classes}
            row = []
            for sp, clips in splits.items():
                r = score(clips, thr, args.tol).get(cls) or {}
                row.append(f"{sp} P {r.get('tag_precision')!s:6} R {r.get('tag_recall')!s:6} n {r.get('hyp', 0):4}")
                curves.setdefault(cls, {}).setdefault(sp, []).append(
                    [round(float(t), 2), r.get("tag_precision"), r.get("tag_recall"), r.get("hyp", 0)])
            print(f"  t {t:.2f}  " + "   ".join(row))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"tol_words": args.tol, "columns": ["threshold", "tag_precision",
                                                                          "tag_recall", "tags"],
                                        "curves": curves}, indent=1))


if __name__ == "__main__":
    main()
