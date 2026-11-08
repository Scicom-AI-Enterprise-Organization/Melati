#!/usr/bin/env python3
"""Build the TRAINING corpus, disjoint from the published benchmark. RUNS ON THE BOX.

The published dataset is the benchmark (test-only). Training data is a separate build, and
disjointness here is STRUCTURAL rather than a list we have to trust:

  | pool             | benchmark uses        | training uses              |
  |------------------|-----------------------|----------------------------|
  | FSD50K           | `eval` split          | `dev` split (35,676 clips) |
  | FMA              | shards 0-1            | shards 2-12                |
  | Malaysian speech | sarawak/manglish/fleurs | Malaysian-Emilia (unused by benchmark) |
  | silence          | seed 0                | seed 1000, different floors|
  | reduplication    | seed 0, units A       | seed 1000, held-out units  |

`benchmark/exclusions.json` is then applied as a second, belt-and-braces check, so an
overlap has to defeat both mechanisms to reach the training set.

Emits train.jsonl / val.jsonl only. There is no test split here: the benchmark IS the test
set, and re-deriving one would just invite drift.

Targets:
  blank label ("")        non-speech -> teach the model to emit nothing
  true transcript          speech    -> anti-forgetting, keeps WER from moving
  exact repeat count       reduplication -> teach it to stop after n repeats

  python scripts/build_training_corpus.py --nonspeech-hours 105 --positive-hours 60
"""
import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCL = ROOT / "benchmark" / "exclusions.json"


def load_exclusions() -> dict:
    if not EXCL.exists():
        sys.exit(f"missing {EXCL}: run scripts/freeze_benchmark.py first")
    d = json.loads(EXCL.read_text())
    return {k: set(v) for k, v in d["exclusions"].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "corpus")
    ap.add_argument("--nonspeech-hours", type=float, default=105.0)
    ap.add_argument("--positive-hours", type=float, default=60.0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1000, help="must differ from the benchmark's seed 0")
    ap.add_argument("--audio-root", type=Path, default=ROOT / "audio_train",
                    help="where the training audio is staged (NOT audio/, which is the benchmark)")
    args = ap.parse_args()
    if args.seed == 0:
        sys.exit("seed 0 is the benchmark's seed; use a different one")

    excl = load_exclusions()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows, stats = [], defaultdict(lambda: [0, 0.0])
    dropped = defaultdict(int)

    for man in sorted(args.audio_root.glob("*/manifest.csv")):
        arm = man.parent.name
        for r in csv.DictReader(man.open(encoding="utf-8")):
            # Belt-and-braces: reject anything whose source item is in the benchmark.
            keys = [
                ("fma_tracks", f"{r.get('artist','')}|{r.get('title','')}"),
                ("fsd50k_ids", r.get("fsd50k_id", "")),
                ("lingualibre_stems", Path(r["audio_filepath"]).stem),
                (f"{r.get('source','')}_stems", Path(r["audio_filepath"]).stem),
            ]
            if any(v and v in excl.get(k, ()) for k, v in keys):
                dropped[arm] += 1
                continue

            blank = arm in ("silence", "music", "nonspeech")
            text = "" if blank else (r.get("reference_text") or r.get("text") or "")
            if not blank and not text.strip():
                dropped[arm] += 1
                continue
            dur = float(r.get("duration_s") or 0)
            rows.append({
                "audio_filepath": str((man.parent / r["audio_filepath"]).resolve()),
                "arm": arm, "text": text, "is_blank_label": blank,
                "duration_s": dur, "lang": r.get("lang", ""), "license": r.get("license", ""),
            })
            stats[arm][0] += 1
            stats[arm][1] += dur / 3600

    if not rows:
        print(f"nothing staged under {args.audio_root}.")
        print("Stage training audio there first (on the box), e.g.:")
        print("  build_music_arm.py      --shards 2-12   --out audio_train/music")
        print("  build_nonspeech_arm.py  --split dev     --out audio_train/nonspeech")
        print("  mine_emilia.py          --phrases       --out audio_train/emilia")
        print("  make_silence_arm.py     --seed 1000     --out audio_train/silence")
        return

    # Cap each pool at its target so the blank/positive ratio is deliberate, not accidental.
    blank_rows = [r for r in rows if r["is_blank_label"]]
    pos_rows = [r for r in rows if not r["is_blank_label"]]
    rng.shuffle(blank_rows); rng.shuffle(pos_rows)

    def take(pool, hours):
        out, acc = [], 0.0
        for r in pool:
            if acc >= hours * 3600:
                break
            out.append(r); acc += r["duration_s"]
        return out, acc / 3600

    blank_rows, bh = take(blank_rows, args.nonspeech_hours)
    pos_rows, ph = take(pos_rows, args.positive_hours)
    kept = blank_rows + pos_rows
    rng.shuffle(kept)

    nval = max(1, int(len(kept) * args.val_frac))
    splits = {"val": kept[:nval], "train": kept[nval:]}
    for sp, recs in splits.items():
        f = args.out / f"{sp}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = sum(r["duration_s"] for r in recs) / 3600
        nb = sum(r["is_blank_label"] for r in recs)
        print(f"{sp:<6} {len(recs):>7} rows  {h:>7.2f} h  blank {nb:>6} ({nb/max(len(recs),1):.0%}) -> {f.name}")

    print(f"\nblank-label {bh:.1f} h (target {args.nonspeech_hours:.0f})"
          f" | positive {ph:.1f} h (target {args.positive_hours:.0f})")
    if dropped:
        print("dropped by benchmark-exclusion / empty text:", dict(dropped))
    print("\nper-arm staged:")
    for a, (n, h) in sorted(stats.items()):
        print(f"  {a:<20}{n:>7} clips {h:>8.2f} h")


if __name__ == "__main__":
    main()
