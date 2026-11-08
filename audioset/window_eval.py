#!/usr/bin/env python3
"""Where in a clip Melati's clip tagger hears each sound, read off short windows. A released checkpoint
(audioset/release.py builds from it) is run over windows of AudioSet's strongly labelled test clips, as its card says
to run a short clip (its filterbanks padded to 1024 frames); each 160 ms step then takes the windows that cover it,
by their maximum, mean or minimum. Frame mAP as audioset/frames_eval.py scores PretrainedSED and the frame head: each
class's average precision over every 160 ms of every clip, averaged over the classes PretrainedSED has and the clips
label. The whole clip's probability on every step (--win 10) is the floor: the timing is what windows add over it.
RUNS ON THE BOX, in .venv-sed: one process a GPU over a slice of the clips, then --merge.

    python audioset/window_eval.py --ckpt $MELATI_DATA/models/audioset/frames1_fw0/clip.pt --shard 0/5
    python audioset/window_eval.py --ckpt $MELATI_DATA/models/audioset/frames1_fw0/clip.pt --merge \\
        --out $MELATI_DATA/results/audioset/window_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import ROOT, SR, classes, load  # noqa: E402
from frames_train import STEP_S, STEPS, frame_targets  # noqa: E402

WINS = (1.0, 2.0, 4.0, 10.0)      # seconds; 10 is the whole clip, one window
HOP_S = 0.32
AGGS = ("max", "mean", "min")


def starts(n, win, hop):
    """Window starts (samples) over a clip of n samples: every hop, and a last one that ends with the clip."""
    return [0] if n <= win else list(range(0, n - win, hop)) + [n - win]


_COVER = {}


def cover(st, win):
    """For each 160 ms step, the windows whose span holds the step's centre (the nearest window if none does)."""
    key = (tuple(st), win)
    if key not in _COVER:
        out = []
        for k in range(STEPS):
            t = (k + 0.5) * STEP_S * SR
            ws = [j for j, s in enumerate(st) if s <= t < s + win]
            out.append(ws or [int(np.argmin([abs(s + win / 2 - t) for s in st]))])
        _COVER[key] = out
    return _COVER[key]


def shard(a):
    import zoo
    i, n = map(int, a.shard.split("/"))
    W, _, lens, _ = load("strong_test")
    import torch
    m = zoo.load(f"melati:{a.ckpt}")
    out = Path(a.work) / f"{a.ckpt.parent.name}"
    out.mkdir(parents=True, exist_ok=True)
    rows = range(i, len(W), n)
    for win_s in WINS:
        f = out / f"win{win_s:g}_{i}of{n}.npy"
        if f.exists():
            continue
        win, hop = int(win_s * SR), int(HOP_S * SR)
        aggs = AGGS if win_s < 10 else AGGS[:1]            # one window: every way of pooling it is the same
        P = np.zeros((len(rows), STEPS, 527, len(aggs)), np.float16)
        clips, owner = [], []

        def flush():
            probs = m([np.asarray(c, np.float32) for c in clips])
            for (r, k, st), p in zip(owner, np.split(probs, np.cumsum([len(o[2]) for o in owner])[:-1])):
                for s_, ws in enumerate(cover(st, win)):
                    q = p[ws]
                    for j, agg in enumerate(aggs):
                        P[r, s_, :, j] = getattr(q, agg)(0)
            clips.clear(); owner.clear()
        for r, k in enumerate(rows):
            n_ = max(int(lens[k]), win) if win_s < 10 else W.shape[1]
            st = starts(min(n_, W.shape[1]), win, hop)
            clips += [W[k, s:s + win] for s in st]
            owner.append((r, k, st))
            if len(clips) >= a.batch:
                flush()
        if clips:
            flush()
        np.save(f, P)
        print(f"[window] {win_s:g} s: {len(rows)} clips -> {f}", flush=True)
        torch.cuda.empty_cache()


def merge(a):
    from frames_eval import frame_map, pretrainedsed
    W, _, _, _ = load("strong_test")
    sm = np.load(ROOT / "cache" / "strong_test_meta.npz")
    index = {mid: i for i, mid, _ in classes()}
    FY = np.stack([frame_targets(json.loads(s) or [], index) for s in sm["strong"]]).astype(np.uint8)
    base = pretrainedsed(W)
    shared = [c for c in range(527) if not np.isnan(base[0, 0, c]) and FY[:, :, c].any()]
    labelled = [c for c in range(527) if FY[:, :, c].any()]
    d = Path(a.work) / a.ckpt.parent.name
    n = len(list(d.glob("win10_*of*.npy")))
    res = {"model": f"melati:{a.ckpt}", "clips": int(len(W)), "classes_shared": len(shared),
           "classes_labelled": len(labelled), "hop_s": HOP_S, "step_s": STEP_S,
           "pretrainedsed": {"shared": round(frame_map(base, FY, shared), 4)}, "windows": {}}
    best, best_p = None, None
    for win_s in WINS:
        parts = [np.load(d / f"win{win_s:g}_{i}of{n}.npy", mmap_mode="r") for i in range(n)]
        P = np.zeros((len(W),) + parts[0].shape[1:], np.float16)
        for i, p in enumerate(parts):
            P[i::n] = p
        for j, agg in enumerate(AGGS if win_s < 10 else AGGS[:1]):
            key = f"{win_s:g}s/{agg}" if win_s < 10 else "clip"
            res["windows"][key] = {"shared": round(frame_map(P[..., j], FY, shared), 4)}
            print(f"[window] {key}: frame mAP {res['windows'][key]['shared']:.4f} on the {len(shared)} shared "
                  f"classes (PretrainedSED {res['pretrainedsed']['shared']:.4f})", flush=True)
            if win_s < 10 and (best is None or res["windows"][key]["shared"] > res["windows"][best]["shared"]):
                best, best_p = key, np.ascontiguousarray(P[..., j])
    res["best"] = best
    res["windows"][best]["labelled"] = round(frame_map(best_p, FY, labelled), 4)
    keep = ROOT / "probs" / f"strong_test_{a.ckpt.parent.name}_{best.replace('/', '_')}.npy"
    np.save(keep, best_p)                                  # for the per-class thresholds an inline tag needs
    res["best_probs"] = str(keep)
    fe = ROOT.parent / "results" / "audioset" / "frames_eval_frames1.json"
    if fe.exists():
        r = json.loads(fe.read_text())
        res["frame_head"] = next(v for k, v in r.items() if k.startswith("/"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[window] -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th clip from the i-th")
    ap.add_argument("--batch", type=int, default=96, help="windows a forward pass")
    ap.add_argument("--work", default=str(ROOT / "probs" / "windows"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    merge(a) if a.merge else shard(a)


if __name__ == "__main__":
    main()
