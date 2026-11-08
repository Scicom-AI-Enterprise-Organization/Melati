#!/usr/bin/env python3
"""Which tagger hears Melati's short segments better, and how a short segment should be fed to it. AudioSet's strongly
labelled test clips (every event timed) cut to the lengths a podcast segment runs to (1 to 8 s, one cut a clip at a
seeded random place) and, for reference, whole; each cut scored against the events it holds, for
Melati-AudioSet-Ensemble and v4 (one of its members), with the tagger's 10 s window filled three ways:

  zeros   the cut, then digital silence: how the ensemble was scored on AudioSet and how the sounds stage first fed it
  repeat  the cut over again until the window is full
  native  the cut as it is: each member's own features of it, an EAT model's filterbanks padded to its 1024 frames
          (how serve/sounds_server.py feeds v4)

An event counts for a cut that holds MIN_OVERLAP of it (all of it, if shorter); one it barely touches is left out of
that class's score for that cut. A class is scored at a length where at least MIN_POS cuts hold it. RUNS ON THE BOX,
in .venv-sed (deploy/box_audioset_short_eval.sh).

    python audioset/short_eval.py run --slice 0/5          # one GPU's share of the clips, every cut and filling
    python audioset/short_eval.py score --out $MELATI_DATA/results/audioset/short_eval.json
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import ROOT, SR, classes, load  # noqa: E402

REPO, REV = "Scicom-intl/Melati-AudioSet-Ensemble", "b3175658f59fea64ecddee98661ec1dabb0f59a5"
LENGTHS = (1, 2, 3, 5, 8, 10)                # seconds; 10 is the whole clip, as AudioSet is scored
FILLS = ("zeros", "repeat", "native")
TAGGERS = ("ensemble", "v4")
WINDOW = 10 * SR
MIN_OVERLAP, MIN_POS = 0.25, 10
OUT = ROOT / "probs" / "short"
S = {}                                       # shared with the scoring workers by fork


def fills(length):
    return ("zeros",) if length == 10 else FILLS


def starts(lens):
    """{length: each clip's cut start in samples, -1 where the clip is shorter than the cut}; the whole clip at 10."""
    out = {}
    for n in LENGTHS:
        room = np.asarray(lens, np.int64) - n * SR
        u = np.random.default_rng(n).random(len(lens))
        out[n] = np.zeros(len(lens), np.int64) if n == 10 else np.where(room >= 0, (u * np.maximum(room, 0)).astype(np.int64), -1)
    return out


def run(a):
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    i, n = map(int, a.slice.split("/"))
    W, _, lens, _ = load("strong_test")
    st = starts(lens)
    path = snapshot_download(REPO, revision=REV)
    sys.path.insert(0, path)
    from melati_ensemble import Ensemble
    ens = Ensemble(path, device="cuda").bf16_weights()
    mine = np.arange(len(W))[i::n]
    OUT.mkdir(parents=True, exist_ok=True)
    for length in LENGTHS:
        for how in fills(length):
            f = OUT / f"{length}s_{how}_{i}of{n}.npz"
            if f.exists():
                continue
            keep = [k for k in mine if st[length][k] >= 0]
            got = {t: [] for t in TAGGERS}
            for b in range(0, len(keep), a.batch):
                ks = keep[b:b + a.batch]
                # the whole clip is the cached row: 10 s, a shorter clip already padded with zeros
                x = np.stack([W[k] if length == 10 else W[k, st[length][k]:st[length][k] + length * SR] for k in ks])
                x = torch.from_numpy(x.astype(np.float32)).cuda()
                if how == "zeros":
                    x = F.pad(x, (0, WINDOW - x.shape[1]))
                elif how == "repeat":
                    x = x.repeat(1, math.ceil(WINDOW / x.shape[1]))[:, :WINDOW]
                per = ens.forward(x)
                got["ensemble"].append(torch.stack(list(per.values())).mean(0).cpu().numpy().astype(np.float16))
                got["v4"].append(per["melati-v4"].cpu().numpy().astype(np.float16))
            np.savez(f, clips=np.asarray(keep), **{t: np.concatenate(v) for t, v in got.items()})
            print(f"[short] {length} s, {how}: {len(keep)} cuts -> {f.name}", flush=True)
    print("SHORT_RUN_DONE", flush=True)


def targets(length, clips, lens, st, events, index):
    """(cuts, 527): 1 where the cut holds an event of the class, -1 where it only touches one (left out), else 0."""
    Y = np.zeros((len(clips), 527), np.int8)
    for r, k in enumerate(clips):
        s0 = st[length][k] / SR
        s1 = lens[k] / SR if length == 10 else s0 + length
        for a, b, mid in events[k]:
            c = index.get(mid)
            if c is None or b <= a:
                continue
            ov = min(b, s1) - max(a, s0)
            if ov >= min(MIN_OVERLAP, b - a) - 1e-6:
                Y[r, c] = 1
            elif ov > 0 and Y[r, c] == 0:
                Y[r, c] = -1
    return Y


def ap(job):
    from sklearn.metrics import average_precision_score
    key, c = job
    y, p = S["Y"][key[0]][:, c], S["P"][key][:, c]
    m = y >= 0
    return average_precision_score(y[m], p[m])


def at(y, p, t):
    m = y >= 0
    y, hit = y[m] == 1, p[m] >= t
    return {"precision": round(float((y & hit).sum() / max(hit.sum(), 1)), 4),
            "recall": round(float((y & hit).sum() / max(y.sum(), 1)), 4), "cuts_over": int(hit.sum())}


def score(a):
    _, _, lens, _ = load("strong_test")
    sm = np.load(ROOT / "cache" / "strong_test_meta.npz")
    events = [json.loads(s) or [] for s in sm["strong"]]
    index = {mid: i for i, mid, _ in classes()}
    names = [n for _, _, n in classes()]
    st = starts(lens)
    S["Y"], S["P"], cols = {}, {}, {}
    for length in LENGTHS:
        for how in fills(length):
            parts = [np.load(f) for f in sorted(OUT.glob(f"{length}s_{how}_*of*.npz"))]
            clips = np.concatenate([p["clips"] for p in parts])
            order = np.argsort(clips)
            for t in TAGGERS:
                S["P"][(length, how, t)] = np.concatenate([p[t] for p in parts])[order].astype(np.float32)
        S["Y"][length] = Y = targets(length, clips[order], lens, st, events, index)
        cols[length] = [c for c in range(527) if (Y[:, c] == 1).sum() >= MIN_POS and (Y[:, c] == 0).any()]
    keys = list(S["P"])
    with Pool(a.workers) as pool:
        aps = dict(zip(keys, (pool.map(ap, [(k, c) for c in cols[k[0]]]) for k in keys)))
    music, silence = names.index("Music"), names.index("Silence")
    res = {"repo": f"{REPO}@{REV}", "min_overlap_s": MIN_OVERLAP, "min_positive_cuts": MIN_POS, "lengths": {}}
    for length in LENGTHS:
        Y = S["Y"][length]
        e = res["lengths"][f"{length} s"] = {"cuts": int(len(Y)), "classes": len(cols[length]), "map": {}, "music": {},
                                              "silence": {}}
        for how in fills(length):
            for t in TAGGERS:
                k, name = (length, how, t), f"{t}, {how}"
                p = S["P"][k]
                e["map"][name] = round(float(np.mean(aps[k])), 4)
                e["music"][name] = {f"at {g}": at(Y[:, music], p[:, music], g) for g in (0.5, 0.8)}
                e["silence"][name] = {"share_p0.1": round(float((p[:, silence] >= 0.1).mean()), 4),
                                      "share_p0.5": round(float((p[:, silence] >= 0.5).mean()), 4)}
        print(f"[short] {length} s: {len(Y)} cuts, {len(cols[length])} classes | "
              + " | ".join(f"{k} {v:.4f}" for k, v in e["map"].items()), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[short] -> {a.out}", flush=True)


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap_.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--slice", default="0/1", help="i/n: every n-th clip from the i-th")
    p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("score")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=48)
    a = ap_.parse_args()
    (run if a.cmd == "run" else score)(a)


if __name__ == "__main__":
    main()
