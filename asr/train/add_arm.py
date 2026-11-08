#!/usr/bin/env python3
"""A new mix = an existing mix + one more replay arm, without rebuilding (or touching) the others.

build_mix.py writes every mix from one seeded pass, so rerunning it to add an arm would rewrite the
mixes runs are training from. This reads one finished mix, appends an arm from a manifest in the CV22
arm's format (audio_filepath relative to the manifest, text, lang, duration_s), shuffles with a fixed
seed, writes the new mix beside it and adds its entry to the mixes' summary.json.

    python asr/train/add_arm.py --base $MELATI_DATA/asr/mixes/all_cv22_r25.jsonl \\
        --arm $MELATI_DATA/asr/audio_train/tts44k/manifest.csv --name tts44k --out all_cv22_r25_tts
"""
import argparse, csv, json, random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--name", required=True, help="the arm's name in the mix rows")
    ap.add_argument("--out", required=True, help="new mix name, written beside --base")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.base.open(encoding="utf-8")]
    extra = [{"audio_filepath": str((args.arm.parent / r["audio_filepath"]).resolve()), "text": r["text"],
              "lang": r["lang"], "arm": args.name, "is_blank": False, "duration_s": float(r["duration_s"])}
             for r in csv.DictReader(args.arm.open(encoding="utf-8")) if (r.get("text") or "").strip()]
    rows += extra
    random.Random(args.seed).shuffle(rows)
    out = args.base.parent / f"{args.out}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    blanks = sum(r["is_blank"] for r in rows)
    stats = {"clips": len(rows), "blank_share": round(blanks / len(rows), 4),
             "hours": round(sum(r["duration_s"] for r in rows) / 3600, 2),
             "languages": len({r["lang"] for r in rows if r["lang"]}), "arms": dict(Counter(r["arm"] for r in rows)),
             "derived_from": args.base.stem, "added_arm": args.name}
    summ = args.base.parent / "summary.json"
    s = json.loads(summ.read_text()) if summ.exists() else {}
    s[args.out] = stats
    summ.write_text(json.dumps(s, indent=1))
    print(f"{out.name}: {len(rows)} clips ({len(extra)} from {args.name}), {stats['hours']} h, "
          f"{stats['languages']} languages")


if __name__ == "__main__":
    main()
