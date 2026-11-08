#!/usr/bin/env python3
"""A non-verbal head's tag precision and recall at every threshold, per class, on NonverbalTTS's calibration and test
clips (the pipeline's candidates, eval_nonverbal_gaps.score's rule). nonverbal/gap_rule.py eval reports only the
lowest threshold reaching the target; this shows why a class got none. RUNS ON THE BOX.

    python scripts/gap_curve.py --heads $MELATI_DATA/models/nv_heads_rel5 --classes cough laugh breath
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nonverbal"))
from gap_rule import load  # noqa: E402
from melati import nv_gaps as NG  # noqa: E402
from scripts.eval_nonverbal_gaps import score  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=Path, required=True)
ap.add_argument("--classes", nargs="+", default=["cough", "laugh", "breath"])
ap.add_argument("--cache", type=Path, default=Path(f"{MELATI_DATA}/nv/nvtts_align.json"))
ap.add_argument("--out", type=Path, default=None, help="also write the curves as JSON")
args = ap.parse_args()
GRID = (0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98)
curves = {}
from melati.ser import HeadNV  # noqa: E402
head = HeadNV("whisper", heads_dir=str(args.heads))
for which in ("calib", "test"):
    clips = load(args.cache, which)
    found, cl_clips = [], []
    for cl, x in clips:
        cands = NG.candidates(x, cl["words"])
        found.append(cands)
        cl_clips += [NG.clip_of(x, c) for c in cands]
    P = np.concatenate([head(cl_clips[s:s + 64])[0] for s in range(0, len(cl_clips), 64)])
    k, scored = 0, []
    for (cl, _), cands in zip(clips, found):
        rows = [dict(c, dist=dict(zip(NG.HEAD_LABELS, map(float, P[k + i])))) for i, c in enumerate(cands)]
        k += len(cands)
        scored.append({"aligned": True, "tags": [tuple(t) for t in cl["tags"]], "cands": rows})
    for cls in args.classes:
        line = []
        for t in GRID:
            r = score(scored, {c: (t if c == cls else 2.0) for c in set(NG.TO_CLASS.values())}, 1).get(cls) or {}
            line.append(f"{t}: {r.get('hyp', 0)} tags {r.get('tag_precision')}/{r.get('tag_recall')}")
            curves.setdefault(which, {}).setdefault(cls, {})[str(t)] = {f: r.get(f) for f in
                                                                       ("hyp", "tag_precision", "tag_recall", "ref")}
        print(f"{which} {cls} (ref {sum(1 for s in scored for c, _ in s['tags'] if c == cls)}): " + " | ".join(line),
              flush=True)
if args.out:
    args.out.write_text(json.dumps({"heads": str(args.heads), "curves": curves}, indent=1))
