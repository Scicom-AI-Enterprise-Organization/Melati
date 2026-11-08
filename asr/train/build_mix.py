#!/usr/bin/env python3
"""Build the Melati ASR training mixes: the hallucination recipe, with and without replay.

The base is the `all` mix of "How to Reduce Whisper Hallucination" (arXiv 2609.32560), whose
best checkpoint (LoRA r64, lr 5e-4, 1000 steps) took silence hallucination from 0.619 to 0.024
and phrase recovery from 0.698 to 0.827, and cost FLEURS macro CER 0.305 -> 0.398:

  corpus         non-speech with EMPTY targets (FSD50K dev, FMA shards 2-12, synthetic floors),
                 plus real Malaysian speech mined from Malaysian-Emilia
  lexicon_synth  synthesised hallucination phrases that ARE spoken (the contrastive half)
  wild           real voice-free audio that made a model hallucinate, empty targets

Replay adds Common Voice 22 (`asr/scripts/build_cv22.py`): validated speech in every language
Whisper has a token for, sampled evenly per language. Its share is counted in clips, because a
clip is what fills a batch slot:

  all            the paper's mix, rebuilt here. The no-replay control.
  all_cv22_r25   all + CV22 so that CV22 is 25% of the clips
  all_cv22_r50   all + CV22 so that CV22 is 50% of the clips
  cv22_only      CV22 alone: what replay does with no hallucination recipe at all

Every mix shares the same `all` rows, so the pair (all, all_cv22_rXX) differs ONLY in the
replay clips. CV22's own test sample never enters a mix.

    python asr/train/build_mix.py --corpus $MELATI_DATA/asr/corpus/train.jsonl \\
        --cv22 $MELATI_DATA/asr/audio_train/cv22/train/manifest.csv \\
        --cache $MELATI_DATA/asr/hub --out $MELATI_DATA/asr/mixes
"""
import argparse, csv, json, random
from collections import Counter
from pathlib import Path

REPO = "Scicom-intl/Whisper-Hallucination"


def load_corpus(path: Path):
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        rows.append({"audio_filepath": r["audio_filepath"], "text": r.get("text", ""),
                     "lang": r.get("lang") or "", "arm": r.get("arm", ""),
                     "is_blank": bool(r.get("is_blank_label")),
                     "duration_s": float(r.get("duration_s") or 0)})
    return rows


def materialise(config: str, split: str, cache: Path, keep):
    """One config/split of the published benchmark, written to FLAC files once.

    The trainer wants paths, not bytes, and the sweep reads the mix many times. `keep(row)`
    returns the training row for a dataset row, or None to skip it."""
    from datasets import Audio, load_dataset

    out = cache / f"{config}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    manifest, done = out / "manifest.jsonl", out / ".complete"
    if done.exists() and manifest.exists():
        rows = [json.loads(l) for l in manifest.open(encoding="utf-8")]
        print(f"{config}/{split}: {len(rows)} clips already materialised at {out}")
        return rows
    ds = load_dataset(REPO, config, split=split).cast_column("audio", Audio(decode=False))
    rows = []
    with manifest.open("w", encoding="utf-8") as fh:
        for r in ds:
            row = keep(r)
            if row is None:
                continue
            path = out / f"{r['id']}.flac"
            if not path.exists():
                path.write_bytes(r["audio"]["bytes"])
            row["audio_filepath"] = str(path)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    done.touch()
    print(f"{config}/{split}: materialised {len(rows)} clips -> {out}")
    return rows


def synth_row(r):
    return {"text": r["phrase"], "lang": r["lang"], "arm": "lexicon_synth", "is_blank": False,
            "duration_s": float(r.get("duration_s") or 0)}


def wild_row(r):
    """Wild `train` is real audio a VAD confirms has no speech: the target is empty. Anything
    in it that carries a transcript is not a blank clip and is left out, as in the paper."""
    if (r.get("reference_text") or "").strip():
        return None
    return {"text": "", "lang": "", "arm": "wild", "is_blank": True,
            "duration_s": float(r.get("duration_s") or 0)}


def load_cv22(path: Path):
    rows = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        rows.append({"audio_filepath": str((path.parent / r["audio_filepath"]).resolve()),
                     "text": r["text"], "lang": r["lang"], "arm": "cv22", "is_blank": False,
                     "duration_s": float(r["duration_s"])})
    return rows


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    blanks = sum(r["is_blank"] for r in rows)
    hrs = sum(r["duration_s"] for r in rows) / 3600
    langs = len({r["lang"] for r in rows if r["lang"]})
    print(f"{path.name:<20} {len(rows):>7} clips  {blanks / max(len(rows), 1):>5.0%} blank  "
          f"{hrs:>6.1f} h  {langs:>3} langs  arms={dict(Counter(r['arm'] for r in rows).most_common(6))}")
    return {"clips": len(rows), "blank_share": round(blanks / max(len(rows), 1), 4),
            "hours": round(hrs, 2), "languages": langs,
            "arms": dict(Counter(r["arm"] for r in rows))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cv22", type=Path, required=True, help="CV22 train manifest.csv")
    ap.add_argument("--cache", type=Path, required=True, help="where hub splits are materialised")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = load_corpus(args.corpus)
    synth = materialise("lexicon_synth", "train", args.cache, synth_row)
    wild = materialise("wild", "train", args.cache, wild_row)
    cv22 = load_cv22(args.cv22)
    print(f"corpus: {sum(r['is_blank'] for r in corpus)} blank + "
          f"{sum(not r['is_blank'] for r in corpus)} positive | lexicon_synth {len(synth)} | "
          f"wild {len(wild)} | cv22 {len(cv22)}")

    def shuffled(xs):
        xs = list(xs); rng.shuffle(xs); return xs

    base = corpus + synth + wild
    summary = {"all": write(args.out / "all.jsonl", shuffled(base))}
    for pct in (25, 50):
        want = round(len(base) * pct / (100 - pct))
        if want <= len(cv22):
            replay = rng.sample(cv22, want)
        else:
            print(f"!! r{pct} wants {want} CV22 clips, the pool has {len(cv22)}: cycling it")
            replay = (cv22 * (want // len(cv22) + 1))[:want]
        summary[f"all_cv22_r{pct}"] = write(args.out / f"all_cv22_r{pct}.jsonl", shuffled(base + replay))
    summary["cv22_only"] = write(args.out / "cv22_only.jsonl", shuffled(cv22))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
