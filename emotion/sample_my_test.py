#!/usr/bin/env python3
"""Pick the clips for Melati's Malaysian emotion test set from an export. RUNS ON THE BOX.

Two strata, both recorded on every clip so an analysis can weight them:
  random     uniform over exported segments: what share of podcast speech carries each emotion (the priors the
             thresholds were chosen with were assumptions; this measures them)
  pred:<x>   segments the current tagger calls <x>, most confident first: so the rare emotions have enough
             positives to measure a precision at all

Clips are 1.5 to 15 s (long enough to judge, short enough to label quickly), at most --per-recording from any one
recording, so a single talkative show cannot fill the set.

    python emotion/sample_my_test.py --export $MELATI_DATA/pilot/export_rel --out $MELATI_DATA/emo/my_test/sample.jsonl \\
        --summary $MELATI_DATA/results/emotion/my_test_sample.json
"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--random", type=int, default=400)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--per-recording", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", type=Path, help="also write the counts per stratum and language here (the paper's)")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    rows = []
    for f in sorted((args.export / "manifest").glob("*.parquet")):
        rows += pq.read_table(f, columns=["id", "recording", "speaker", "duration", "language", "text",
                                          "emotion_segment", "emotion_segment_p"]).to_pylist()
    rows = [r for r in rows if 1.5 <= (r["duration"] or 0) <= 15.0 and (r["text"] or "").strip()]
    rng = random.Random(args.seed)
    per_rec = Counter()
    picked, seen = [], set()

    def take(r, stratum):
        if r["id"] in seen or per_rec[r["recording"]] >= args.per_recording:
            return False
        audio = args.export / "audio" / r["recording"] / f"{r['id']}.flac"
        if not audio.exists():
            return False
        seen.add(r["id"]); per_rec[r["recording"]] += 1
        picked.append({"id": r["id"], "audio": str(audio), "text": r["text"], "lang": r["language"],
                       "recording": r["recording"], "speaker": r["speaker"], "duration": r["duration"],
                       "stratum": stratum, "pred": r["emotion_segment"], "pred_p": r["emotion_segment_p"]})
        return True

    pool = rows[:]
    rng.shuffle(pool)
    n = 0
    for r in pool:
        if n >= args.random:
            break
        n += take(r, "random")
    by = defaultdict(list)
    for r in rows:
        if r["emotion_segment"] and r["emotion_segment"] != "neutral":
            by[r["emotion_segment"]].append(r)
    for lab in LABELS[1:]:
        k = 0
        for r in sorted(by[lab], key=lambda r: -(r["emotion_segment_p"] or 0)):
            if k >= args.per_class:
                break
            k += take(r, f"pred:{lab}")
    rng.shuffle(picked)                       # annotators must not see the strata in order
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in picked:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    strata, langs = dict(Counter(p["stratum"] for p in picked)), dict(Counter(p["lang"] for p in picked))
    print(f"{len(picked)} clips from {len(per_rec)} recordings -> {args.out}")
    print(strata, langs)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({"clips": len(picked), "random": strata.get("random", 0), "strata": strata,
                                            "languages": langs, "recordings": len(per_rec),
                                            "export": str(args.export)}))


if __name__ == "__main__":
    main()
