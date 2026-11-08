#!/usr/bin/env python3
"""Melati's non-verbal head on AMI's held-out meetings: laughs and coughs in conversation. RUNS ON THE BOX.

The candidates are the pipeline's (melati.nv_gaps.candidates, clip_of) in the gaps between anyone's words in each
meeting's headset mix: all of them, marked or not. The head labels them and melati.nv_gaps.decide makes the tags.
A tag is right where an AMI mark of its class lies within 0.1 s of its candidate; a mark is found where a tag of
its class does. Two sets of thresholds: NonverbalTTS's (what ships: chosen on its train clips by
nonverbal/gap_rule.py eval) and AMI's own, chosen the same way (the lowest threshold reaching the target tag
precision) on the dev meetings, which says what the head does in meetings once calibrated there.

AMI marks no ordinary breathing, so breath tags are counted, not scored. A mark inside someone's words (a laugh
over another speaker, or a speech-laugh) can have no candidate; those are counted apart (`in_speech`), and the
rest of the misses split into no candidate at all (the gap finder) and a candidate the head did not tag.

    python nonverbal/eval_ami.py --heads $MELATI_DATA/models/nv_heads_rel3 \\
        --thresholds $MELATI_DATA/results/nonverbal/gap_eval_rel3.json \\
        --out $MELATI_DATA/results/nonverbal/ami_eval_rel3.json
"""
import argparse, bisect, json, os, sys, zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ami import ANNOTATIONS, meetings, read, split_of, union, wav_path  # noqa: E402
from candidates import windows  # noqa: E402
from melati import nv_gaps as NG  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

SCORED = ("laugh", "cough")
TOL = 0.1


def meeting(root: Path, z, m: str, rule: dict):
    """-> (candidates [{s, e}], their clips, marks [{label, s, e, in_speech}]) for one meeting."""
    words, sounds = read(z, m)
    x, sr = sf.read(str(wav_path(root, m)), dtype="float32", always_2d=True)
    x = x.mean(1)
    assert sr == NG.SR, sr
    spans = union(words)
    cands, clips = [], []
    for a, b, run in windows(spans, len(x) / sr):
        xw = x[int(a * sr):int(b * sr)]
        for c in NG.candidates(xw, [{"s": w["s"] - a, "e": w["e"] - a} for w in run], **rule):
            cands.append({"s": a + c["s"], "e": a + c["e"]})
            clips.append(NG.clip_of(xw, c))
    starts = [s["s"] for s in spans]
    marks = []
    for v in sounds:
        t = (v["s"] + max(v["e"], v["s"])) / 2
        i = bisect.bisect_right(starts, t) - 1
        marks.append({"label": v["label"], "s": v["s"], "e": max(v["e"], v["s"]),
                      "in_speech": bool(i >= 0 and spans[i]["e"] > t)})
    return cands, clips, marks


def near(a, m):
    return a["s"] <= m["e"] + TOL and a["e"] >= m["s"] - TOL


def top_nv(row):
    nv = {NG.TO_CLASS[k]: float(v) for k, v in zip(NG.HEAD_LABELS, row) if k in NG.TO_CLASS}
    c = max(nv, key=nv.get)
    return c, nv[c]


def score(data, thresholds):
    """Per class over a set of meetings: marks, where they are, what was tagged, precision and recall."""
    out = {}
    for c in SCORED + ("breath",):
        r = {"marks": 0, "in_speech": 0, "with_candidate": 0, "no_candidate_in_speech": 0, "found": 0, "tags": 0,
             "right": 0}
        t = thresholds.get(c)
        for cands, tops, marks in data:
            tags = [cd for cd, (k, p) in zip(cands, tops) if k == c and t is not None and p >= t]
            r["tags"] += len(tags)
            mk = [m for m in marks if m["label"] == c]
            r["right"] += sum(any(near(g, m) for m in mk) for g in tags)
            for m in mk:
                r["marks"] += 1
                r["in_speech"] += m["in_speech"]
                has = any(near(cd, m) for cd in cands)
                r["with_candidate"] += has
                r["no_candidate_in_speech"] += (not has) and m["in_speech"]
                r["found"] += any(near(g, m) for g in tags)
        if c in SCORED:
            r["precision"] = round(r["right"] / r["tags"], 4) if r["tags"] else None
            r["recall"] = round(r["found"] / r["marks"], 4) if r["marks"] else None
            out_of_speech = r["marks"] - r["in_speech"]
            r["recall_outside_speech"] = round(r["found"] / out_of_speech, 4) if out_of_speech else None
        else:
            r = {"tags": r["tags"]}                    # AMI does not mark breathing: nothing to score against
        out[c] = r
    return out


def calibrate(data, target=0.8, min_hyp=5):
    """Per scored class, the lowest threshold whose tag precision reaches `target` (scripts/eval_nonverbal_gaps.py's
    rule, on AMI's dev meetings)."""
    out = {}
    for c in SCORED:
        out[c] = None
        for t in np.arange(0.05, 0.96, 0.05):
            r = score(data, {c: round(float(t), 2)})[c]
            if r["tags"] < min_hyp:
                break
            if r["precision"] >= target:
                out[c] = {"threshold": round(float(t), 2), "precision": r["precision"], "recall": r["recall"],
                          "tags": r["tags"]}
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ami", type=Path, default=Path(f"{MELATI_DATA}/nv/raw/ami"))
    ap.add_argument("--heads", type=Path, required=True, help="a directory holding <model>.pt (melati.ser.HeadNV)")
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--thresholds", type=Path, required=True, help="nonverbal/gap_rule.py eval's output")
    ap.add_argument("--rule", default='{"floor_db": 2}')
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from melati.ser import HeadNV
    ev = json.loads(args.thresholds.read_text())
    shipped = {c: v["threshold"] for c, v in ev[args.rule]["thresholds"].items() if v}
    head = HeadNV(args.model, heads_dir=str(args.heads))
    data = {"validation": [], "test": []}
    with zipfile.ZipFile(args.ami / ANNOTATIONS) as z:
        for m in meetings(z):
            sp = split_of(m)
            if sp not in data or not wav_path(args.ami, m).exists():
                continue
            cands, clips, marks = meeting(args.ami, z, m, json.loads(args.rule))
            P = np.concatenate([head(clips[s:s + args.batch])[0] for s in range(0, len(clips), args.batch)]) \
                if clips else np.zeros((0, len(NG.HEAD_LABELS)), np.float32)
            data[sp].append((cands, [top_nv(row) for row in P], marks))
            print(f"  {m} ({sp}): {len(cands)} candidates, {len(marks)} marks", flush=True)
    own = calibrate(data["validation"], args.target)
    own_thr = {c: v["threshold"] for c, v in own.items() if v}
    res = {"heads": str(args.heads), "rule": json.loads(args.rule), "tol_s": TOL, "target_precision": args.target,
           "meetings": {sp: len(d) for sp, d in data.items()},
           "candidates": {sp: sum(len(c) for c, _, _ in d) for sp, d in data.items()},
           "nonverbaltts_thresholds": shipped, "ami_thresholds": own,
           "test": {"nonverbaltts_thresholds": score(data["test"], shipped),
                    "ami_thresholds": score(data["test"], own_thr)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    for k, v in res["test"].items():
        print(k, json.dumps({c: (r.get("precision"), r.get("recall"), r.get("marks"), r.get("in_speech"), r["tags"])
                             for c, r in v.items()}), flush=True)
    print("AMI_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
