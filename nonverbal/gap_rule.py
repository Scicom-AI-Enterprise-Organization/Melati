#!/usr/bin/env python3
"""Why the gap finder misses breaths, and what a different rule finds instead. RUNS ON THE BOX.

On the aligned NonverbalTTS clips (nonverbal/align_cache.py), with the pipeline's own candidate rule
(melati.nv_gaps.candidates) under any settings:

  diagnose   for every reference tag with no candidate within one word boundary: is there a gap at all once the
             words are dilated, is the sound in it below the level threshold (and by how much), or too brief
  sweep      candidate recall per class and candidates per clip, for each rule given as JSON keyword arguments
  eval       each rule's candidates labelled by the non-verbal head, thresholds chosen on the calibration draw as
             scripts/eval_nonverbal_gaps.py chooses them, scored there and on the test split: what a rule is
             worth after the labeller, not before it (a rule is picked on the calibration numbers alone)

    python nonverbal/gap_rule.py diagnose --cache $MELATI_DATA/nv/nvtts_align.json
    python nonverbal/gap_rule.py sweep --cache ... --rules '{}' '{"floor_db": 3}'
    python nonverbal/gap_rule.py eval --cache ... --rules '{}' '{"floor_db": 3}' --model whisper
"""
import argparse, io, json, os, random, sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from melati import nv_gaps as NG  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

FRAME = NG.FRAME


def load(cache: Path, which: str, calib_n=800):
    """(clip, audio) for the test split, the calibration draw, or the training clips outside it."""
    from datasets import Audio, load_dataset
    c = json.loads(cache.read_text())
    split = "test" if which == "test" else "train"
    d = load_dataset("deepvk/NonverbalTTS", split=split).cast_column("audio", Audio(decode=False))
    keep = None
    if split == "train":
        idx = list(range(len(d)))
        random.Random(0).shuffle(idx)
        calib = set(idx[:calib_n])
        keep = calib if which == "calib" else set(range(len(d))) - calib
    out = []
    for cl in c[split]:
        if not cl["aligned"] or (keep is not None and cl["row"] not in keep):
            continue
        r = d[cl["row"]]
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        out.append((cl, A.resample(x.mean(1), sr, NG.SR)))
    return out


def gap_stats(x, words, slot, rel_db=NG.REL_DB, floor_db=NG.FLOOR_DB, min_active=NG.MIN_ACTIVE, dilate=0.03, **_):
    """What the default rule sees in gap `slot`: its length after dilation, its loudest frame against the speech
    level, and the level threshold, all in dB relative to the speech level."""
    n = int(FRAME * NG.SR)
    m = len(x) // n
    db = 10 * np.log10(np.mean(x[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    inside = np.zeros(m, bool)
    for w in words:
        inside[max(0, int(w["s"] / FRAME)):min(m, int(w["e"] / FRAME) + 1)] = True
    speech = float(np.median(db[inside])) if inside.any() else float(np.max(db))
    floor = float(np.percentile(db, 10))
    thr = max(floor + floor_db, speech + rel_db)
    dur = len(x) / NG.SR
    a = words[slot - 1]["e"] if slot > 0 else 0.0
    b = words[slot]["s"] if slot < len(words) else dur
    a2, b2 = a + (dilate if slot > 0 else 0.0), b - (dilate if slot < len(words) else 0.0)
    fa, fb = int(a2 / FRAME), int(b2 / FRAME)
    seg = db[fa:fb] if fb > fa else np.array([])
    return {"gap_s": round(b - a, 3), "gap_dilated_s": round(max(0.0, b2 - a2), 3),
            "max_rel": round(float(seg.max() - speech), 1) if len(seg) else None,
            "thr_rel": round(thr - speech, 1), "floor_rel": round(floor - speech, 1),
            "active_s": round(float((seg >= thr).sum()) * FRAME, 3) if len(seg) else 0.0,
            "thr_by": "floor" if floor + floor_db >= speech + rel_db else "speech"}


def has_cand(cands, slot, tol=1):
    return any(abs(c["slot"] - slot) <= tol for c in cands)


def diagnose(clips, cls="breath", rule=None):
    rule = rule or {}
    min_active = rule.get("min_active", NG.MIN_ACTIVE)
    why, rows = Counter(), []
    for cl, x in clips:
        cands = NG.candidates(x, cl["words"], **rule)
        for c, slot in cl["tags"]:
            if c != cls:
                continue
            st = gap_stats(x, cl["words"], slot, **rule)
            if has_cand(cands, slot):
                why["found"] += 1
                st["found"] = True
            else:
                st["found"] = False
                if st["gap_dilated_s"] < min_active:
                    why["no gap"] += 1
                elif st["max_rel"] is None or st["max_rel"] < st["thr_rel"]:
                    why["too quiet"] += 1
                elif st["active_s"] < min_active:
                    why["too brief"] += 1
                else:
                    why["other"] += 1
            rows.append(st)
    return why, rows


def sweep(clips, rules):
    out = {}
    for rule in rules:
        kw = json.loads(rule)
        found, ref, n = Counter(), Counter(), 0
        for cl, x in clips:
            cands = NG.candidates(x, cl["words"], **kw)
            n += len(cands)
            for c, slot in cl["tags"]:
                ref[c] += 1
                found[c] += has_cand(cands, slot)
        out[rule] = {"candidates": n, "per_clip": round(n / max(len(clips), 1), 2),
                     "candidate_recall": {c: round(found[c] / ref[c], 3) for c in sorted(ref) if ref[c] >= 10}}
        print(rule, json.dumps(out[rule]), flush=True)
    return out


def evaluate(cal, test, rules, model="whisper", target=0.8, tol=1, batch=64):
    """-> {rule: {thresholds, calib, test}}; a candidate is labelled once even when several rules find it."""
    from melati.ser import HeadNV
    from scripts.eval_nonverbal_gaps import calibrate, score
    sets = {"calib": cal, "test": test}
    found = {}                                   # (set, clip index, rule) -> candidates
    keys, clips = {}, []
    for rule in rules:
        kw = json.loads(rule)
        for name, cs in sets.items():
            for i, (cl, x) in enumerate(cs):
                cands = NG.candidates(x, cl["words"], **kw)
                found[(name, i, rule)] = cands
                for c in cands:
                    k = (name, i, c["s"], c["e"])
                    if k not in keys:
                        keys[k] = len(clips)
                        clips.append(NG.clip_of(x, c))
    print(f"{len(clips)} distinct candidates over {len(rules)} rules", flush=True)
    head = HeadNV(model)
    P = np.concatenate([head(clips[s:s + batch])[0] for s in range(0, len(clips), batch)])
    out = {}
    for rule in rules:
        scored = {}
        for name, cs in sets.items():
            scored[name] = [{"aligned": True, "tags": [tuple(t) for t in cl["tags"]],
                             "cands": [dict(c, dist=dict(zip(NG.HEAD_LABELS, map(float, P[keys[(name, i, c["s"], c["e"])]]))))
                                       for c in found[(name, i, rule)]]}
                            for i, (cl, _) in enumerate(cs)]
        thr = calibrate(scored["calib"], target, tol)
        use = {c: v["threshold"] for c, v in thr.items() if v}
        out[rule] = {"thresholds": thr, "calib": score(scored["calib"], use, tol), "test": score(scored["test"], use, tol)}
        brief = {sp: {c: (out[rule][sp].get(c, {}).get("tag_precision"), out[rule][sp].get(c, {}).get("tag_recall"))
                      for c in ("breath", "laugh", "cough")} for sp in ("calib", "test")}
        print(rule, "thresholds", use, json.dumps(brief), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["diagnose", "sweep", "eval"])
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--cache", type=Path, default=Path(f"{MELATI_DATA}/nv/nvtts_align.json"))
    ap.add_argument("--which", default="test", choices=["test", "calib", "rest"])
    ap.add_argument("--rules", nargs="*", default=["{}"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.step == "eval":
        res = evaluate(load(args.cache, "calib"), load(args.cache, "test"), args.rules, args.model)
        if args.out:
            args.out.write_text(json.dumps(res, indent=1))
        return
    clips = load(args.cache, args.which)
    print(f"{len(clips)} aligned clips ({args.which})", flush=True)
    if args.step == "diagnose":
        res = {"rule": json.loads(args.rules[0])}
        for cls in ("breath", "laugh"):
            why, rows = diagnose(clips, cls, res["rule"])
            miss = [r for r in rows if not r["found"]]
            quiet = [r["thr_rel"] - r["max_rel"] for r in miss if r["max_rel"] is not None and r["max_rel"] < r["thr_rel"]]
            res[cls] = {"why": dict(why.most_common()), "missed_gap_s": [r["gap_s"] for r in miss],
                        "quiet_margin_db": quiet, "rows": rows}
            print(cls, dict(why.most_common()), flush=True)
            if quiet:
                print(f"  too quiet by (dB): p25 {np.percentile(quiet, 25):.1f}  p50 {np.median(quiet):.1f}  "
                      f"p75 {np.percentile(quiet, 75):.1f}", flush=True)
            gaps = [r["gap_s"] for r in miss]
            if gaps:
                print(f"  missed tags' gap length (s): p25 {np.percentile(gaps, 25):.2f}  p50 {np.median(gaps):.2f}  "
                      f"p75 {np.percentile(gaps, 75):.2f};  share under 0.14 s: "
                      f"{np.mean(np.array(gaps) < 0.14):.2f}", flush=True)
    else:
        res = sweep(clips, args.rules)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
