#!/usr/bin/env python3
"""The frame detector on NonverbalTTS at every threshold: tags (by the slot each event starts in), how many are right
within one word, and how many of the reference tags are found; with the share of its events over speech. A
diagnostic for nonverbal/frames.py eval, which only reports the lowest threshold reaching the target, and the source
of the out-of-domain numbers at the thresholds that ship (--out). RUNS ON THE BOX.

    python scripts/frames_diag.py --model $MELATI_DATA/models/nv_frames3/whisper.pt --out .../frames3_nvtts.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nonverbal"))
import frames as FR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--feats", type=Path, default=FR.FEATS)
ap.add_argument("--classes", nargs="+", default=["laugh", "cough"])
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()
import torch  # noqa: E402
ck = torch.load(args.model, map_location="cpu")
head = FR.frame_head().cuda()
head.load_state_dict(ck["state"])
res = {"model": str(args.model)}
for w in ("calib", "test"):
    meta = json.loads((args.feats / "nvtts" / f"{w}.json").read_text())
    P = FR.predict(head, [np.load(args.feats / "nvtts" / f"{w}.npy", mmap_mode="r")])[0]
    for cls in args.classes:
        k = FR.CLASSES.index(cls)
        probs = [p[:m["frames"], k] for p, m in zip(P, meta)]
        peak = np.array([p.max() if len(p) else 0 for p in probs])
        has = np.array([any(c == cls for c, _ in m["tags"]) for m in meta])
        print(f"{w} {cls}: {len(meta)} clips, {has.sum()} with one; clip peak p: with median "
              f"{np.median(peak[has]) if has.any() else float('nan'):.3f}, without {np.median(peak[~has]):.3f}", flush=True)
        for t in FR.GRID[-10:]:
            hyps = [FR.nvtts_tags(p, m["words"], t) for p, m in zip(probs, meta)]
            over = [FR.nvtts_tags(p, m["words"], t, over_speech_only=True) for p, m in zip(probs, meta)]
            s, o = FR.score_slots(meta, hyps, cls), FR.score_slots(meta, over, cls)
            res.setdefault(w, {}).setdefault(cls, {})[str(t)] = s
            print(f"  t={t}: tags {s['hyp']} right {s['right']} precision {s['precision']} recall {s['recall']} | "
                  f"over speech {o['hyp']} ({o['precision']}) | clips with p>=t: {(peak >= t).sum()}, "
                  f"{((peak >= t) & has).sum()} have one", flush=True)
if args.out:
    args.out.write_text(json.dumps(res, indent=1))
