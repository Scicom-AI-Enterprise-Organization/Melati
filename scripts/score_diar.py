#!/usr/bin/env python3
"""Diarization error rate of Stage B against reference turns (frame-level, 10 ms, optimal mapping).

DER = (missed speech + false alarm + speaker confusion) / reference speech, the standard
definition with overlap counted per speaker; a collar removes +-collar seconds around every
reference boundary from scoring. Used on the VoxConverse dev set to check the stage end to end;
the diarization model's own paper reports it properly.

    python scripts/score_diar.py --work $MELATI_DATA/dev/vox/work2 --refs $MELATI_DATA/dev/vox
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

STEP = 0.01


def frames(turns, n):
    spk = sorted({t[2] for t in turns})
    m = np.zeros((len(spk), n), bool)
    for s, e, k in turns:
        m[spk.index(k), int(s / STEP):int(e / STEP)] = True
    return m


def der(ref, hyp, collar=0.0):
    n = int(max([e for _, e, _ in ref] + [e for _, e, _ in hyp] + [0]) / STEP) + 1
    R, H = frames(ref, n), frames(hyp, n)
    keep = np.ones(n, bool)
    if collar > 0:
        c = int(collar / STEP)
        for s, e, _ in ref:
            for t in (int(s / STEP), int(e / STEP)):
                keep[max(0, t - c):t + c] = False
    R, H = R[:, keep], H[:, keep]
    if len(H) == 0:
        return float(R.sum(0).sum() > 0), {}
    ov = R.astype(np.int32) @ H.T.astype(np.int32)
    ri, hi = linear_sum_assignment(-ov)
    nr, nh = R.sum(0), H.sum(0)
    correct = np.zeros(R.shape[1], np.int32)
    for a, b in zip(ri, hi):
        correct += (R[a] & H[b])
    miss = np.maximum(0, nr - nh).sum()
    fa = np.maximum(0, nh - nr).sum()
    conf = (np.minimum(nr, nh) - correct).sum()
    tot = max(nr.sum(), 1)
    return (miss + fa + conf) / tot, {"miss": miss / tot, "fa": fa / tot, "conf": conf / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows, tot = {}, {"ref_s": 0.0, "err0": 0.0, "err25": 0.0}
    for d in sorted(p for p in args.work.iterdir() if (p / "diar.json").exists()):
        ref = json.loads((args.refs / f"{d.name}.ref.json").read_text())["turns"]
        hyp = json.loads((d / "diar.json").read_text())["turns"]
        d0, parts = der(ref, hyp)
        d25, _ = der(ref, hyp, 0.25)
        rs = sum(e - s for s, e, _ in ref)
        rows[d.name] = {"der": round(d0 * 100, 2), "der_c025": round(d25 * 100, 2),
                        **{k: round(v * 100, 2) for k, v in parts.items()},
                        "ref_speakers": len({k for *_, k in ref}), "hyp_speakers": len({k for *_, k in hyp})}
        tot["ref_s"] += rs; tot["err0"] += d0 * rs; tot["err25"] += d25 * rs
        print(d.name, rows[d.name])
    summary = {"der": round(100 * tot["err0"] / max(tot["ref_s"], 1e-9), 2),
               "der_c025": round(100 * tot["err25"] / max(tot["ref_s"], 1e-9), 2), "recordings": len(rows)}
    print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "recordings": rows}, indent=1))


if __name__ == "__main__":
    main()
