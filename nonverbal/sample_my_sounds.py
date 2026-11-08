#!/usr/bin/env python3
"""Pick the clips for Melati's Malaysian non-verbal test set from a run's finished recordings. RUNS ON THE BOX.

The dataset's own tags (melati.pipeline.export_rows, exactly what the published rows hold), sampled per kind:
breaths, laughs, coughs, fillers, and controls (the middle of a word gap where nothing is tagged). Each clip is the
tagged moment with --context seconds either side, from the exported segment's own audio. Annotators hear it with the
moment marked and say what is there, never told what was tagged, and the transcript they see has the fillers taken
out; a kind's precision on Malay speech is how often they hear that kind (nonverbal/score_my_sounds.py). At most
--per-recording clips of a kind come from one recording, so a few talkative shows cannot fill the set.

    python nonverbal/sample_my_sounds.py --sources $MELATI_DATA/podcast/sources.jsonl \\
        --work $MELATI_DATA/podcast/work --out $MELATI_DATA/nv/my_sounds \\
        --summary $MELATI_DATA/results/nonverbal/my_sounds_sample.json
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 24000
KINDS = ("breathes", "laughs", "coughs", "filler", "control")


def candidates(job):
    """Every tagged moment of one recording (no audio decoded): (kind, segment id, start, end, p, source)."""
    from melati.pipeline import export_rows
    work, rec = job
    out = []
    for row, _ in export_rows(Path(work), rec, {}, audio=False):
        ann = json.loads(row["annotation"])
        words = ann.get("words") or []
        evs = ann.get("events") or []
        base = {"seg": row["id"], "recording": row["recording"], "speaker": row["speaker"], "lang": row["language"],
                "duration": row["duration"]}
        for e in evs:
            if e["kind"] in KINDS:
                out.append(dict(base, kind=e["kind"], s=e["start"], e=e["end"], p=e["conf"], src=e["source"]))
        for w in words:
            if w["filler"]:
                out.append(dict(base, kind="filler", s=w["start"], e=w["end"], p=w["conf"], src="word"))
        speech = [w for w in words if not w["filler"]]
        for a, b in zip(speech, speech[1:]):                  # a control: an untagged word gap of 0.4 s or more
            if b["start"] - a["end"] < 0.4:
                continue
            m = (a["end"] + b["start"]) / 2
            if any(x["start"] - 0.5 < m < x["end"] + 0.5 for x in evs + [w for w in words if w["filler"]]):
                continue
            out.append(dict(base, kind="control", s=round(m - 0.15, 3), e=round(m + 0.15, 3), p=None, src="gap"))
    return out


def cut(job):
    """The clips of one recording: its picked moments with context, as FLAC, and the transcript around each."""
    import numpy as np
    import soundfile as sf
    from melati.pipeline import export_rows
    work, rec, picks, out, ctx = job
    by_seg = defaultdict(list)
    for c in picks:
        by_seg[c["seg"]].append(c)
    done = []
    for row, x in export_rows(Path(work), rec, {}):
        for c in by_seg.get(row["id"], []):
            a, b = max(0.0, c["s"] - ctx), min(len(x) / SR, c["e"] + ctx)
            cid = f"{c['seg']}_{c['kind']}_{int(c['s'] * 1000)}"
            path = Path(out) / "clips" / f"{cid}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.clip(x[int(a * SR):int(b * SR)], -1, 1), SR, format="FLAC", subtype="PCM_16")
            words = [w for w in json.loads(row["annotation"])["words"] if not w["filler"]
                     and w["end"] > a and w["start"] < b]
            text = " ".join(([w["text"] for w in words if w["start"] < c["s"]] + ["▮"] +
                             [w["text"] for w in words if w["start"] >= c["s"]]))
            done.append(dict(c, id=cid, audio=str(path), mark=[round(c["s"] - a, 3), round(c["e"] - a, 3)],
                             clip_s=round(b - a, 3), text=text))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-kind", type=int, default=150)
    ap.add_argument("--controls", type=int, default=60)
    ap.add_argument("--per-recording", type=int, default=2)
    ap.add_argument("--context", type=float, default=1.5)
    ap.add_argument("--max-recordings", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in args.sources.open()]
    recs = [r for r in recs if (args.work / r["id"] / "fill.jsonl").exists()]
    rng.shuffle(recs)
    recs = recs[: args.max_recordings]
    with ProcessPoolExecutor(args.workers) as ex:
        cands = [c for cs in ex.map(candidates, [(str(args.work), r) for r in recs], chunksize=8) for c in cs]
    print(f"{len(recs)} finished recordings, candidates: {dict(Counter(c['kind'] for c in cands))}", flush=True)
    rng.shuffle(cands)
    quota = {k: (args.controls if k == "control" else args.per_kind) for k in KINDS}
    per = Counter()
    picked = []
    for c in cands:
        k = c["kind"]
        if quota[k] > 0 and per[(c["recording"], k)] < args.per_recording:
            picked.append(c); quota[k] -= 1; per[(c["recording"], k)] += 1
    by_rec = defaultdict(list)
    for c in picked:
        by_rec[c["recording"]].append(c)
    rec_of = {r["id"]: r for r in recs}
    with ProcessPoolExecutor(args.workers) as ex:
        items = [i for its in ex.map(cut, [(str(args.work), rec_of[r], cs, str(args.out), args.context)
                                           for r, cs in by_rec.items()]) for i in its]
    rng.shuffle(items)                       # annotators must not meet the kinds in order
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "sample.jsonl").open("w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")
    kinds, langs = dict(Counter(i["kind"] for i in items)), dict(Counter(i["lang"] for i in items))
    print(f"{len(items)} clips from {len(by_rec)} recordings -> {args.out / 'sample.jsonl'}\n{kinds} {langs}")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({"clips": len(items), "kinds": kinds, "languages": langs,
                                            "recordings": len(by_rec), "recordings_scanned": len(recs),
                                            "candidates": dict(Counter(c["kind"] for c in cands))}, indent=1))


if __name__ == "__main__":
    main()
