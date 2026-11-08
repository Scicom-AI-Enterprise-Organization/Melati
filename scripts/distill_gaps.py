#!/usr/bin/env python3
"""Where the released emotion and non-verbal models fall short, class by class, and which model does that class
best: the candidates a distillation would learn from. Emotion: every model and combination in ensemble_all.json
against the release (its own probe), on the same held-out clips, as recall at 75% precision re-weighted to podcast
rates (thresholds chosen on one half, measured on the other) and held-out recall. Non-verbal: the numbers its card
reads (nonverbal/release.py Numbers). RUNS ON THE BOX.

    python scripts/distill_gaps.py
"""
import os
import json
import sys
from pathlib import Path

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = Path(f"{MELATI_DATA}/results/emotion")
LABELS = ["happy", "sad", "angry", "surprised", "fearful"]
# trained only on the release's licence-clean corpora (emotion/release.py TRAIN): the release itself; every other
# head saw the excluded ones, and MERaLiON-SER and emotion2vec+ are third-party models trained on EmoBox-type corpora
CLEAN = {"release"}


def cv(r, lab):
    v = ((r.get("thresholds_cv") or {}).get(lab) or {})
    return v.get("recall")


def main():
    ens = json.loads((R / "ensemble_all.json").read_text())["results"]
    from emotion.release import numbers            # the release's own figures, as its card computes them
    rel, _, _, cv_rel, _, _ = numbers()
    rows = {"release": {"uar": rel["held_out"]["all"]["uar"], **{l: (cv_rel.get(l) or {}).get("recall") for l in LABELS}}}
    for k, r in ens.items():
        rows[k] = {"uar": r["held_out"]["uar"], **{l: cv(r, l) for l in LABELS}}
    singles = [k for k in rows if " + " not in k]
    print("EMOTION, recall at 75% precision (podcast rates), held-out clips")
    print(f"{'model':62s} {'UAR':>6s} " + " ".join(f"{l:>9s}" for l in LABELS))
    show = ["release"] + [k for k in singles if k != "release"] + ["meralion + e2v",
                                                                   "whisper + whisper_ft + w2vbert + wavlm + e2v + meralion + e2v"]
    fmt = lambda v: "     -   " if v is None else f"{100 * v:8.1f}%"  # noqa: E731
    for k in show:
        if k in rows:
            r = rows[k]
            print(f"{k[:62]:62s} {r['uar']:6.3f} " + " ".join(fmt(r[l]) for l in LABELS))
    print("\nbest per class (single model / any combination) against the release:")
    for l in LABELS:
        bs = max(singles, key=lambda k: rows[k][l] or -1)
        bc = max(rows, key=lambda k: rows[k][l] or -1)
        print(f"  {l:9s} release {fmt(rows['release'][l])}  best single {bs} {fmt(rows[bs][l])}  "
              f"best combination {bc[:60]} {fmt(rows[bc][l])}")

    from nonverbal.release import Numbers
    N = Numbers("rel4", 3, "rel4f")
    print("\nNON-VERBAL (found / precise where labelled), PodcastFillers test through the pipeline")
    for c, name in (("laughter", "laughs"), ("breath", "breaths")):
        a, b, q = (N.pf_class(d, c) for d in (N.pf, N.pf_prev, N.pf_omni))
        print(f"  {name:8s} release {a}  previous release (NonverbalTTS in training) {b}  Qwen3-Omni {q}")
    print("AMI test (found, precise)")
    for c in ("laugh", "cough"):
        print(f"  {c:8s} release {N.ami_frames(c)}  previous {N.ami_prev[c]}")
    print("NonverbalTTS test (found, precise)")
    for c, now in (("laugh", N.nv_frames("laugh")), ("breath", N.nv_breath()), ("cough", N.nv_frames("cough"))):
        p = N.nv_prev[c]
        print(f"  {c:8s} release {now}  previous {(p['tag_recall'], p['tag_precision'])}")


if __name__ == "__main__":
    main()
