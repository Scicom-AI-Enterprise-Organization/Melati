#!/usr/bin/env python3
"""What should place an inline sound tag in time. Frame mAP on AudioSet's strongly labelled test clips, as
audioset/frames_eval.py scores it, for PretrainedSED's frame taggers, the frame head trained beside v3
(audioset/frames_train.py), Melati's clip tagger (v4) read off 2 s windows (audioset/window_eval.py), and each
frame-level one weighted by v4's probability for the whole clip: v4 says which sounds are in a clip, the frame model
where. Then, for the best of them, each class's lowest threshold with 0.8 frame precision, the recall there and the
positives it rests on. RUNS ON THE BOX, in .venv-sed: the frame head on one GPU (cached), the rest arithmetic.

    python audioset/timing_eval.py --out $MELATI_DATA/results/audioset/timing_eval.json
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "nonverbal")]
from data import ROOT, classes, load  # noqa: E402
from frames_eval import melati, pretrainedsed  # noqa: E402
from frames_train import frame_targets  # noqa: E402

PROBS = ROOT / "probs"
MODELS = ROOT.parent / "models" / "audioset"
HEAD = MODELS / "frames1" / "best.pt"            # v3's frame model: the encoder, its clip head and its frame head
V4 = "frames1_fw0"                               # audioset/window_eval.py's run folder for v4
PRECISION = 0.8
S = {}                                           # the arrays, shared with the workers by fork


def timing():
    """{name: function of a class index -> its scores over every 160 ms step of every clip}."""
    psed, head, clip, win = S["psed"], S["head"], S["clip"], S["win"]
    f = lambda a: (lambda c: a[:, :, c].astype(np.float32).ravel())      # noqa: E731
    g = lambda a: (lambda c: (a[:, :, c].astype(np.float32) * clip[:, None, c]).ravel())   # noqa: E731
    return {"pretrainedsed": f(psed), "frame head (v3)": f(head), "v4, 2 s windows": f(win),
            "pretrainedsed x v4 clip": g(psed), "frame head x v4 clip": g(head),
            "mean of the two frame models": (lambda c: ((psed[:, :, c].astype(np.float32)
                                                         + head[:, :, c].astype(np.float32)) / 2).ravel()),
            "mean of the two x v4 clip": (lambda c: ((psed[:, :, c].astype(np.float32) + head[:, :, c].astype(np.float32))
                                                     / 2 * clip[:, None, c]).ravel())}


def ap(job):
    from sklearn.metrics import average_precision_score
    name, c = job
    return average_precision_score(S["fy"][:, :, c].ravel(), timing()[name](c))


def threshold(job, precision=PRECISION):
    """The lowest threshold whose frames are at least `precision` right, with the recall there."""
    from sklearn.metrics import precision_recall_curve
    name, c = job[:2]
    precision = job[2] if len(job) > 2 else precision
    p, r, t = precision_recall_curve(S["fy"][:, :, c].ravel(), timing()[name](c))
    ok = np.flatnonzero(p[:-1] >= precision)
    if not len(ok):
        return None
    k = ok[0]
    return {"threshold": round(float(t[k]), 4), "recall": round(float(r[k]), 4), "precision": round(float(p[k]), 4)}


def arrays():
    """Fill S with the test clips' frame targets and every model's probabilities (the frame head's computed once
    and cached) -> (class names, the classes PretrainedSED has and the clips label, every class they label)."""
    W, _, _, _ = load("strong_test")
    sm = np.load(ROOT / "cache" / "strong_test_meta.npz")
    index = {mid: i for i, mid, _ in classes()}
    S["fy"] = np.stack([frame_targets(json.loads(s) or [], index) for s in sm["strong"]]).astype(np.uint8)
    S["psed"] = pretrainedsed(W)
    cached = PROBS / "strong_test_frames1_head.npy"
    if not cached.exists():
        np.save(cached, melati(W, HEAD))
    S["head"] = np.load(cached)
    d = PROBS / "windows" / V4
    n = len(list(d.glob("win10_*of*.npy")))
    clip = np.zeros((len(W), 527), np.float32)
    for i in range(n):
        clip[i::n] = np.load(d / f"win10_{i}of{n}.npy")[:, 0, :, 0]     # one window: the same on every step
    S["clip"] = clip
    S["win"] = np.load(PROBS / f"strong_test_{V4}_2s_mean.npy")
    fy = S["fy"]
    shared = [c for c in range(527) if not np.isnan(S["psed"][0, 0, c]) and fy[:, :, c].any()]
    labelled = [c for c in range(527) if fy[:, :, c].any()]
    S["psed"] = np.nan_to_num(S["psed"], nan=0.0)
    return [n for _, _, n in classes()], shared, labelled


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--out", type=Path, required=True)
    ap_.add_argument("--workers", type=int, default=48)
    a = ap_.parse_args()
    names, shared, labelled = arrays()
    fy = S["fy"]
    res = {"clips": int(len(fy)), "step_s": 0.16, "classes_shared": len(shared), "classes_labelled": len(labelled),
           "frame_map": {}}
    names_t = list(timing())
    with Pool(a.workers) as pool:
        for name in names_t:
            aps = pool.map(ap, [(name, c) for c in shared])
            res["frame_map"][name] = round(float(np.mean(aps)), 4)
            print(f"[timing] {name}: frame mAP {res['frame_map'][name]:.4f} on the {len(shared)} shared classes",
                  flush=True)
        best = max(res["frame_map"], key=res["frame_map"].get)
        # a frame model that covers every class (the frame head) is also scored where PretrainedSED has nothing
        cols = shared if "pretrainedsed" in best or "two" in best else labelled
        th = pool.map(threshold, [(best, c) for c in cols])
    res["best"] = best
    res["precision"] = PRECISION
    res["classes"] = {names[c]: dict(t or {}, positive_frames=int(fy[:, :, c].sum()),
                                     positive_clips=int(fy[:, :, c].any(1).sum()))
                      for c, t in zip(cols, th)}
    reached = sum(t is not None for t in th)
    print(f"[timing] best: {best}; {reached} of {len(cols)} classes reach {PRECISION} frame precision", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[timing] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
