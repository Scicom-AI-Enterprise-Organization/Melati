#!/usr/bin/env python3
"""What a head's wrong tags of one class are: for every tag of the class on NonverbalTTS's calibration clips that
is not within one word of a reference tag of its class, the reference tags in its clip (class and word distance)
and the head's three most probable labels. RUNS ON THE BOX.

    python scripts/gap_false_tags.py --heads $MELATI_DATA/models/nv_heads_rel6 --cls cough --t 0.75
"""
import argparse, os, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nonverbal"))
from gap_rule import load  # noqa: E402
from melati import nv_gaps as NG  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=Path, required=True)
ap.add_argument("--cls", default="cough")
ap.add_argument("--t", type=float, default=0.75)
ap.add_argument("--which", default="calib")
ap.add_argument("--cache", type=Path, default=Path(f"{MELATI_DATA}/nv/nvtts_align.json"))
args = ap.parse_args()
from melati.ser import HeadNV  # noqa: E402
head = HeadNV("whisper", heads_dir=str(args.heads))
clips = load(args.cache, args.which)
found, xs = [], []
for cl, x in clips:
    cands = NG.candidates(x, cl["words"])
    found.append(cands)
    xs += [NG.clip_of(x, c) for c in cands]
P = np.concatenate([head(xs[s:s + 64])[0] for s in range(0, len(xs), 64)])
k, near, wrong = 0, Counter(), 0
for (cl, x), cands in zip(clips, found):
    for c in cands:
        dist = dict(zip(NG.HEAD_LABELS, map(float, P[k])))
        k += 1
        if NG.decide(dist, {args.cls: args.t}) != args.cls:
            continue
        refs = [(t, s) for t, s in cl["tags"]]
        if any(t == args.cls and abs(s - c["slot"]) <= 1 for t, s in refs):
            continue
        wrong += 1
        close = sorted(((abs(s - c["slot"]), t) for t, s in refs), key=lambda v: v[0])[:3]
        near[close[0][1] if close and close[0][0] <= 1 else "(nothing within a word)"] += 1
        top = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
        print(f"row {cl['row']} slot {c['slot']} {c['e'] - c['s']:.2f}s level {c['level_db']:+.1f} dB | refs {close} | "
              + ", ".join(f"{lab} {p:.2f}" for lab, p in top), flush=True)
print(f"{wrong} wrong {args.cls} tags; the nearest reference tag: {dict(near.most_common())}")
