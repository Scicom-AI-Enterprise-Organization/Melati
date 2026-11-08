#!/usr/bin/env python3
"""Melati's non-verbal heads on the very candidates Qwen3-Omni was scored on, NonverbalTTS. RUNS ON THE BOX.

scripts/eval_nonverbal_gaps.py found the candidates (sound in a word gap of the force-aligned reference words) on
800 NonverbalTTS `train` clips and all 359 `test` clips, asked Qwen3-Omni about each, and saved every candidate with
its times. This asks the heads instead, about the same audio cut the same way, and reuses that script's own
calibration (per class, the lowest threshold giving --target-precision tag precision on `train`) and scoring (a tag
within one word boundary of a reference tag of its class), so the two labellers differ in nothing else.

Each head is scored alone and the heads averaged. Test clips with no candidate still count: their reference tags
are misses (all 359 test clips aligned word for word in that run).

    python nonverbal/eval_gaps.py --candidates $MELATI_DATA/results/nonverbal/nonverbaltts_gaps.json \\
        --models whisper wavlm ast --out $MELATI_DATA/results/nonverbal/nonverbaltts_gaps_heads.json
"""
import argparse, io, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from melati.nv_gaps import HEAD_LABELS, PAD_S, SR, clip_of  # noqa: E402
from scripts.eval_nonverbal import reference  # noqa: E402
from scripts.eval_nonverbal_gaps import calibrate, score, strip  # noqa: E402


def clips_of(split, n, seed=0):
    """NonverbalTTS rows in the order eval_nonverbal_gaps.collect() read them: audio at 16 kHz and the
    reference tags at word boundaries counted over the words that survive stripping."""
    from datasets import Audio, load_dataset
    d = load_dataset("deepvk/NonverbalTTS", split=split).cast_column("audio", Audio(decode=False))
    idx = list(range(len(d)))
    if n and n < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])
    out = []
    for i in idx:
        r = d[i]
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        raw, tags = reference(r["Result"])
        words = [strip(w) for w in raw]
        kept = [sum(1 for w in words[:k] if w) for k in range(len(words) + 1)]
        out.append({"x": A.resample(x.mean(1), sr, SR), "tags": [(cl, kept[sl]) for cl, sl in tags]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["whisper", "wavlm", "ast"])
    ap.add_argument("--target-precision", type=float, default=0.8)
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    saved = json.loads(args.candidates.read_text())
    calib_n = saved["calibration"]["clips"]
    clips = {"train": [], "test": []}
    audio = {"train": clips_of("train", calib_n), "test": clips_of("test", 0)}
    have = {(c["split"], c["clip"]): c for c in saved["candidates"]}
    for split in ("train", "test"):
        for i, a in enumerate(audio[split]):
            c = have.get((split, i))
            if c is None and split == "train":
                continue                        # unknown whether it aligned; it has no candidate either way
            if c is not None:
                assert [tuple(t) for t in c["ref_tags"]] == [tuple(t) for t in a["tags"]], (split, i)
            clips[split].append({"aligned": True, "tags": a["tags"], "x": a["x"],
                                 "cands": [dict(cd) for cd in (c or {}).get("cands", [])]})
    jobs = [(sp, i, k) for sp in clips for i, cl in enumerate(clips[sp]) for k in range(len(cl["cands"]))]
    xs = [clip_of(clips[sp][i]["x"], clips[sp][i]["cands"][k], PAD_S) for sp, i, k in jobs]
    print(f"{len(clips['train'])} calibration clips, {len(clips['test'])} test clips, {len(xs)} candidates", flush=True)

    from melati.ser import HeadNV
    probs = {}
    for m in args.models:
        model = HeadNV(m)
        probs[m] = np.concatenate([model(xs[s:s + args.batch])[0] for s in range(0, len(xs), args.batch)])
        print(f"  {m}: done", flush=True)
        del model
    variants = dict(probs)
    if len(args.models) > 1:
        variants["+".join(args.models)] = np.mean([probs[m] for m in args.models], 0)
    for cl in clips["train"] + clips["test"]:
        cl.pop("x", None)
    out = {"candidates_from": str(args.candidates), "labels": list(HEAD_LABELS), "tol_words": args.tol,
           "target_precision": args.target_precision, "calibration_clips": len(clips["train"]),
           "test_clips": len(clips["test"]), "candidates": len(xs), "variants": {}}
    for name, P in variants.items():
        for (sp, i, k), row in zip(jobs, P):
            clips[sp][i]["cands"][k]["dist"] = {lab: float(v) for lab, v in zip(HEAD_LABELS, row)}
        thr = calibrate(clips["train"], args.target_precision, args.tol)
        use = {c: v["threshold"] for c, v in thr.items() if v}
        out["variants"][name] = {"thresholds": thr, "argmax": score(clips["test"], None, args.tol),
                                 "calibrated": score(clips["test"], use, args.tol)}
        cal = out["variants"][name]["calibrated"]
        print(f"== {name}: thresholds {use}", flush=True)
        for c in ("laugh", "breath", "cough"):
            r = cal.get(c) or {}
            print(f"   {c:7s} tag P {r.get('tag_precision')}  R {r.get('tag_recall')}  "
                  f"(hyp {r.get('hyp')}, ref {r.get('ref')}, candidate R {r.get('candidate_recall')})", flush=True)
    base = saved["test"]["calibrated"]
    out["qwen3_omni"] = {"thresholds": saved["calibration"]["thresholds"], "calibrated": base}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("EVAL_GAPS_DONE", flush=True)


if __name__ == "__main__":
    main()
