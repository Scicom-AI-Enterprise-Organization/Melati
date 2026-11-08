#!/usr/bin/env python3
"""Frame-level taggers on the strongly labelled subset's test clips, on the classes they share: PretrainedSED's
average of BEATs, ATST-F and M2D (the AudioSet tagger Melati's pipeline uses, nonverbal/audioset_pass.py; 447 classes,
mapped onto AudioSet's 527 by name, its 40 ms frames pooled to 160 ms steps by their maximum) against a frame model
audioset/frames_train.py saved. Frame mAP: each class's average precision over every 160 ms of every clip, averaged
over the classes both models have and the test clips label. RUNS ON THE BOX, in .venv-sed.

    python audioset/frames_eval.py --frames $MELATI_DATA/models/audioset/frames1/best.pt \\
        --out $MELATI_DATA/results/audioset/frames_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "nonverbal")]
from data import ROOT, classes, load  # noqa: E402
from frames_train import STEPS, FrameEAT, frame_targets  # noqa: E402
from train import features  # noqa: E402
from zoo import REPOS  # noqa: E402

PROBS = ROOT / "probs"


def pretrainedsed(W, batch=32):
    """PretrainedSED's three, averaged, on 10 s rows -> (N, 64, 527), a class it lacks as NaN."""
    out = PROBS / "strong_test_pretrainedsed.npy"
    if out.exists():
        return np.load(out)
    from audioset_eval import Tagger, class_index
    names = class_index()                                     # their 447 names -> their output index
    ours = {n: i for i, _, n in classes()}
    pairs = [(names[n], ours[n]) for n in names if n in ours]
    tagger = Tagger("BEATs+ATST-F+M2D", batch=batch)
    P = np.full((len(W), STEPS, 527), np.nan, np.float16)
    for i in range(0, len(W), 256):
        for k, p in enumerate(tagger([np.asarray(w, np.float32) for w in W[i:i + 256]])):   # (447, 250) at 25 a second
            p = np.pad(p, ((0, 0), (0, STEPS * 4 - p.shape[1])))             # 10 s -> 10.24 s, as the grid
            steps = p.reshape(p.shape[0], STEPS, 4).max(2)                    # 40 ms frames -> 160 ms steps
            for theirs, j in pairs:
                P[i + k, :, j] = steps[theirs]
    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(out, P)
    return P


@torch.no_grad()
def melati(W, ckpt, batch=48):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    from transformers import AutoModel
    m = FrameEAT(AutoModel.from_pretrained(REPOS[ck["init"]], trust_remote_code=True))
    m.load_state_dict(ck["state"])
    m = m.cuda().eval()
    P = []
    for i in range(0, len(W), batch):
        w = torch.from_numpy(np.asarray(W[i:i + batch], np.float32)).cuda()
        P.append(torch.sigmoid(m(features(w))[1].float()).cpu().numpy().astype(np.float16))
    return np.concatenate(P)


def frame_map(P, FY, cols):
    from sklearn.metrics import average_precision_score
    return float(np.mean([average_precision_score(FY[:, :, c].ravel(), P[:, :, c].astype(np.float32).ravel())
                          for c in cols]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, nargs="*", default=[], help="frame models (audioset/frames_train.py)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    W, _, _, _ = load("strong_test")
    sm = np.load(ROOT / "cache" / "strong_test_meta.npz")
    index = {mid: i for i, mid, _ in classes()}
    FY = np.stack([frame_targets(json.loads(s) or [], index) for s in sm["strong"]]).astype(np.uint8)
    base = pretrainedsed(W)
    shared = [c for c in range(527) if not np.isnan(base[0, 0, c]) and FY[:, :, c].any()]
    labelled = [c for c in range(527) if FY[:, :, c].any()]
    res = {"clips": int(len(W)), "classes_labelled": len(labelled), "classes_shared": len(shared),
           "pretrainedsed": {"shared": round(frame_map(base, FY, shared), 4)}}
    print(f"PretrainedSED (BEATs+ATST-F+M2D): frame mAP {res['pretrainedsed']['shared']:.4f} on {len(shared)} classes",
          flush=True)
    for f in a.frames:
        P = melati(W, f)
        res[str(f)] = {"shared": round(frame_map(P, FY, shared), 4), "labelled": round(frame_map(P, FY, labelled), 4)}
        print(f"{f}: frame mAP {res[str(f)]['shared']:.4f} on the shared classes, {res[str(f)]['labelled']:.4f} on all "
              f"{len(labelled)} labelled", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
