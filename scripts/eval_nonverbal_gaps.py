#!/usr/bin/env python3
"""Where by acoustics, what by an audio LLM: non-verbal tags scored against NonverbalTTS. RUNS ON THE BOX.

Asked to transcribe and tag in one pass, Qwen3-Omni senses that a clip holds breathing but puts
the tag at the wrong word (scripts/eval_nonverbal_llm.py: 13% of its breath tags within one word
of the reference under the prompt that finds most of them) and invents throat clearing. So the
two questions are split:

  where  the reference words are force-aligned (align server), and every gap between two words
         (and before the first, after the last) that holds sound above both the clip's noise floor
         and a level relative to its speech is a CANDIDATE, with its own times. The level is low
         enough to keep a breath (--rel-db), which is far quieter than speech.
  what   each candidate, cut out alone and padded with silence, goes to the `omni` service with a
         closed list of labels (`LABELS`). Speech, a filler and background noise are labels too,
         so "nothing to tag" is an answer it can give.

The labels are asked for as letters (A-P), because a letter is one token: the probabilities of
the first answer token are then a distribution over the list (most label WORDS share a first token,
"c" for cough, cry, click). A class is tagged where its probability clears a per-class threshold,
chosen on a sample of NonverbalTTS `train` as the lowest that reaches --target-precision, and
reported on `test`, which never takes part in a choice. A class with no reference tag in the
calibration sample gets no threshold and is never tagged: its precision cannot be measured.

A candidate at gap k is a tag at word boundary k, the same index NonverbalTTS's inline emoji use.
Scored per class: tag precision and recall at the boundary (within --tol words), clip-level
precision and recall (comparable with eval_nonverbal.py and eval_nonverbal_llm.py), and the
candidate recall: the share of reference tags that had a candidate at their boundary at all,
the ceiling on everything the labeller can do.

    python scripts/eval_nonverbal_gaps.py --endpoints $MELATI_DATA/serve_eval/endpoints.json \\
        $MELATI_DATA/serve_omni/endpoints.json --calib-n 800 --out results/nonverbal/nonverbaltts_gaps.json
"""
import argparse, asyncio, io, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from scripts.eval_nonverbal import reference  # noqa: E402

# the method itself lives in the package, so the pipeline runs exactly what this measures
from melati.nv_gaps import (LABELS, LETTERS, PROMPT, SR, TO_CLASS, candidates, clip_of,  # noqa: E402,F401
                            decide, dist_of)


def strip(w):
    return re.sub(r"^\W+|\W+$", "", w)


async def collect(c, split, n, args, seed=0):
    from datasets import Audio, load_dataset
    d = load_dataset("deepvk/NonverbalTTS", split=split).cast_column("audio", Audio(decode=False))
    idx = list(range(len(d)))
    if n and n < len(idx):
        import random
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])
    clips = []
    for i in idx:
        r = d[i]
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        raw, tags = reference(r["Result"])
        words = [strip(w) for w in raw]
        kept = [sum(1 for w in words[:k] if w) for k in range(len(words) + 1)]
        clips.append({"x": A.resample(x.mean(1), sr, SR), "words": [w for w in words if w],
                      "tags": [(cl, kept[sl]) for cl, sl in tags]})
    al = await c.align([(cl["x"][: 30 * SR], " ".join(cl["words"]), "en") for cl in clips])
    jobs = []
    for i, (cl, a) in enumerate(zip(clips, al)):
        ws = a.get("words") or []
        cl["aligned"] = len(ws) == len(cl["words"]) and len(cl["x"]) <= 30 * SR
        cl["cands"] = []
        if not cl["aligned"]:
            continue
        for cand in candidates(cl["x"], ws, args.rel_db, args.floor_db, args.min_active):
            jobs.append((i, cand, clip_of(cl["x"], cand, args.pad_s)))
    res = await asyncio.gather(*[c.chat_audio("omni", xx, PROMPT, 2, top_logprobs=20) for _, _, xx in jobs])
    for (i, cand, _), (ans, top) in zip(jobs, res):
        cand["answer"], cand["dist"] = ans, dist_of(ans, top)
        clips[i]["cands"].append(cand)
    print(f"[{split}] {sum(cl['aligned'] for cl in clips)}/{len(clips)} clips aligned word for word, "
          f"{len(jobs)} candidates", flush=True)
    for cl in clips:
        cl.pop("x")
    return clips


def score(clips, thresholds, tol):
    st = defaultdict(Counter)
    for cl in clips:
        if not cl["aligned"]:
            continue
        ref, hyp = defaultdict(list), defaultdict(list)
        for cls, slot in cl["tags"]:
            ref[cls].append(slot)
        for cd in cl["cands"]:
            cls = decide(cd["dist"], thresholds)
            if cls:
                hyp[cls].append(cd["slot"])
        cand_slots = [cd["slot"] for cd in cl["cands"]]
        for cls in set(ref) | set(hyp):
            s = st[cls]
            s["ref"] += len(ref[cls])
            s["ref_with_candidate"] += sum(any(abs(r - k) <= tol for k in cand_slots) for r in ref[cls])
            s["ref_found"] += sum(any(abs(r - h) <= tol for h in hyp[cls]) for r in ref[cls])
            s["hyp"] += len(hyp[cls])
            s["hyp_right"] += sum(any(abs(h - r) <= tol for r in ref[cls]) for h in hyp[cls])
            s["clip_tp"] += bool(ref[cls]) and bool(hyp[cls])
            s["clip_fp"] += (not ref[cls]) and bool(hyp[cls])
            s["clip_fn"] += bool(ref[cls]) and not hyp[cls]
    return {cls: {"tag_precision": round(s["hyp_right"] / s["hyp"], 4) if s["hyp"] else None,
                  "tag_recall": round(s["ref_found"] / s["ref"], 4) if s["ref"] else None,
                  "candidate_recall": round(s["ref_with_candidate"] / s["ref"], 4) if s["ref"] else None,
                  "clip_precision": round(s["clip_tp"] / max(s["clip_tp"] + s["clip_fp"], 1), 4),
                  "clip_recall": round(s["clip_tp"] / max(s["clip_tp"] + s["clip_fn"], 1), 4), **dict(s)}
            for cls, s in st.items()}


def calibrate(clips, target, tol, min_hyp=5):
    """Per class, the lowest threshold whose tag precision on these clips reaches `target`."""
    classes = sorted(set(TO_CLASS.values()))
    out = {}
    for cls in classes:
        out[cls] = None
        for t in np.arange(0.05, 0.96, 0.05):
            thr = {c: (round(float(t), 2) if c == cls else 2.0) for c in classes}
            r = score(clips, thr, tol).get(cls) or {}
            if not r.get("ref"):
                break                               # no reference tag of this class: not measurable
            if r.get("hyp", 0) < min_hyp:
                break
            if r["tag_precision"] >= target:
                out[cls] = {"threshold": round(float(t), 2), "precision": r["tag_precision"],
                            "recall": r["tag_recall"], "hyp": r["hyp"], "ref": r["ref"]}
                break
    return out


async def run(args):
    from melati.client import Client, Endpoints
    async with Client(Endpoints(args.endpoints)) as c:
        calib = await collect(c, "train", args.calib_n, args) if args.calib_n else []
        test = await collect(c, "test", 0, args)
    thr = calibrate(calib, args.target_precision, args.tol) if calib else {}
    use = {k: v["threshold"] for k, v in thr.items() if v}
    out = {"model": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "prompt": PROMPT, "tol_words": args.tol,
           "candidate_rule": {"rel_db": args.rel_db, "floor_db": args.floor_db, "min_active_s": args.min_active},
           "calibration": {"split": "train", "clips": len(calib), "target_precision": args.target_precision,
                           "thresholds": thr},
           "test": {"clips": len(test), "aligned": sum(cl["aligned"] for cl in test),
                    "labels": dict(Counter(max(cd["dist"], key=cd["dist"].get) for cl in test for cd in cl["cands"]
                                           if cd["dist"]).most_common()),
                    "argmax": score(test, None, args.tol), "calibrated": score(test, use, args.tol)},
           "candidates": [{"split": sp, "clip": i, "ref_tags": cl["tags"], "words": cl["words"], "cands": cl["cands"]}
                          for sp, cs in (("train", calib), ("test", test)) for i, cl in enumerate(cs) if cl["cands"]]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("thresholds (train):", {k: v and (v["threshold"], v["precision"], v["recall"]) for k, v in thr.items()})
    for name in ("argmax", "calibrated"):
        print(f"== test, {name}")
        for cls, r in sorted(out["test"][name].items()):
            print(f"  {cls:13s} tag P {r['tag_precision']}  R {r['tag_recall']}  cand R {r['candidate_recall']}  "
                  f"clip P {r['clip_precision']} R {r['clip_recall']}  (ref {r['ref']}, hyp {r['hyp']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", nargs="+", type=Path, required=True, help="align + omni (several files merge)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rel-db", type=float, default=-40.0, help="candidate frames: above speech level + this")
    ap.add_argument("--floor-db", type=float, default=6.0, help="...and above the clip's noise floor + this")
    ap.add_argument("--min-active", type=float, default=0.08)
    ap.add_argument("--pad-s", type=float, default=0.3)
    ap.add_argument("--tol", type=int, default=1, help="a tag within this many word boundaries counts")
    ap.add_argument("--calib-n", type=int, default=800, help="NonverbalTTS train clips to choose thresholds on")
    ap.add_argument("--target-precision", type=float, default=0.8)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
