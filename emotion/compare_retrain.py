#!/usr/bin/env python3
"""Retrained emotion heads beside the ones they replace, on the numbers that chose them. RUNS ON THE BOX.

Both folders hold what deploy/box_emo_retrain.sh writes (probe_<encoder>.json, current.json, ensemble_all.json):
--old is the first heads' (the repo's results/emotion, which the box lost on 2026-10-29), --new the retrain's. Per
head: the leave-one-corpus-out mean and the held-out UAR; for the two released classifiers and the seven-model
average the pipeline tags with: held-out UAR, the threshold each label gets, and the recall and precision those
thresholds give when chosen on one half of the held-out clips and measured on the other.

    python emotion/compare_retrain.py --old $MELATI_DATA/v1/results/emotion \\
        --new $MELATI_DATA/results/emotion --out $MELATI_DATA/results/emotion/compare_v1.json
"""
import argparse
import json
from pathlib import Path

ENCODERS = ("whisper", "whisper_ft", "w2vbert", "wavlm", "e2v")
SEVEN = "whisper + whisper_ft + w2vbert + wavlm + e2v + meralion + e2v"


def J(p):
    return json.loads(p.read_text()) if p.exists() else None


def heads(d):
    out = {}
    for enc in ENCODERS:
        r = J(d / f"probe_{enc}.json")
        if r:
            out[enc] = {"loco_mean_uar": r.get("loco_mean_uar"),
                        **{f"held_out_{c}": v["uar"] for c, v in r["held_out"].items()}}
    return out


def average(d):
    out = {}
    cur = J(d / "current.json") or {}
    for name in ("meralion", "e2v", "meralion+e2v"):
        if name in cur:
            out[name] = {"held_out_uar": cur[name]["held_out"]["all"]["uar"],
                         "thresholds": {k: v and v["threshold"] for k, v in cur[name]["thresholds"].items()}}
    ens = (J(d / "ensemble_all.json") or {}).get("results", {}).get(SEVEN)
    if ens:
        out["seven"] = {"held_out_uar": ens["held_out"]["uar"], "held_out_acc": ens["held_out"]["acc"],
                        "thresholds": {k: v and v["threshold"] for k, v in ens["thresholds"].items()},
                        "thresholds_cv": ens["thresholds_cv"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {"old": {"heads": heads(a.old), "average": average(a.old)},
           "new": {"heads": heads(a.new), "average": average(a.new)}}
    a.out.write_text(json.dumps(res, indent=1))
    o, n = res["old"], res["new"]
    for enc in ENCODERS:
        a_, b_ = o["heads"].get(enc, {}), n["heads"].get(enc, {})
        print(f"h_{enc:11s} " + "  ".join(f"{k} {a_.get(k)} -> {b_.get(k)}" for k in sorted(set(a_) | set(b_))))
    for name in ("meralion", "e2v", "meralion+e2v", "seven"):
        a_, b_ = o["average"].get(name, {}), n["average"].get(name, {})
        print(f"{name:13s} held-out UAR {a_.get('held_out_uar')} -> {b_.get('held_out_uar')}  thresholds "
              f"{a_.get('thresholds')} -> {b_.get('thresholds')}")
    for lab, v in (n["average"].get("seven", {}).get("thresholds_cv") or {}).items():
        w = (o["average"].get("seven", {}).get("thresholds_cv") or {}).get(lab)
        print(f"  seven, {lab:9s} recall {w and w['recall']} -> {v and v['recall']}  "
              f"precision {w and w['precision_at_prior']} -> {v and v['precision_at_prior']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
