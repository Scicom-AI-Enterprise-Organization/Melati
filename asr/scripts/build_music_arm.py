#!/usr/bin/env python3
"""Build the ACTUAL-MUSIC arm: real produced music tracks, reference transcript empty.

Distinct from the FSD50K `nonspeech` arm on purpose. FSD50K "Music" is mostly short
instrument samples uploaded to Freesound; what actually plays under a call is a produced
track -- hold music, a YouTube backing bed -- with drums, bass, mixing and often singing.
That is a different acoustic distribution and it is the one that matters operationally.

Source: Free Music Archive, commercial-licence subset, already 16 kHz mono (CC BY 4.0).

Ground truth is the empty string: music is not speech, so a transcript is a hallucination.

VOCALS: FMA carries no instrumental/vocal metadata, so some tracks contain singing. Those
rows are still labelled with an empty reference because the target behaviour for a music
bed is to emit nothing -- but a purist who wants "provably zero words in the audio" should
use the FSD50K `nonspeech` arm instead, where every clip is label-verified voice-free.
The `vocals` column here is honestly recorded as "unknown", not guessed.

  python scripts/build_music_arm.py --tracks 600
"""
import argparse, csv, io, os, random
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
REPO = "benjamin-paine/free-music-archive-commercial-16khz-full"
ALL_SHARDS = [f"data/train-{i:05d}-of-00013.parquet" for i in range(13)]
# Benchmark used shards 0-1; training must use 2-12 so the pools cannot overlap.
SHARDS = ALL_SHARDS[:2]
# Straddle the 30 s window boundary, as in the silence arm.
EXCERPTS = [10.0, 30.0, 45.0]


def decode(raw: bytes) -> np.ndarray | None:
    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        try:                                   # libsndfile without mp3 -> ffmpeg fallback
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as t:
                t.write(raw); src = t.name
            out = src + ".wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src,
                            "-ac", "1", "-ar", str(SR), out], check=True, capture_output=True)
            x, sr = sf.read(out, dtype="float32", always_2d=True)
            os.unlink(src); os.unlink(out)
        except Exception:
            return None
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=int, default=600)
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "music")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", default="0-1",
                    help="FMA shard range, e.g. '0-1' (benchmark) or '2-12' (training)")
    ap.add_argument("--excerpts", nargs="+", type=float, default=None,
                    help="override excerpt lengths in seconds")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="keep source shards; default deletes each after conversion to bound disk")
    args = ap.parse_args()

    global SHARDS, EXCERPTS
    a, _, b = args.shards.partition("-")
    SHARDS = ALL_SHARDS[int(a): int(b or a) + 1]
    if args.excerpts:
        EXCERPTS = args.excerpts

    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    rng = random.Random(args.seed)
    flac_dir = args.out / "flac"; flac_dir.mkdir(parents=True, exist_ok=True)
    rows, made, shard_paths = [], 0, []

    for shard in SHARDS:
        if made >= args.tracks:
            break
        print(f"[fetch] {shard}", flush=True)
        local = hf_hub_download(REPO, shard, repo_type="dataset", token=tok)
        f = pq.ParquetFile(local)
        shard_paths.append(local)
        for batch in f.iter_batches(batch_size=32):
            for rec in batch.to_pylist():
                if made >= args.tracks:
                    break
                au = rec.get("audio")
                if not isinstance(au, dict) or not au.get("bytes"):
                    continue
                x = decode(au["bytes"])
                if x is None or x.size < 12 * SR:        # need room for a real excerpt
                    continue
                want = rng.choice(EXCERPTS)
                n = int(want * SR)
                if x.size <= n:
                    seg = x
                else:                                     # skip the intro; take from inside
                    start = rng.randrange(int(0.15 * x.size), max(int(0.15 * x.size) + 1, x.size - n))
                    seg = x[start:start + n]
                peak = float(np.abs(seg).max())
                if peak <= 0:
                    continue
                seg = np.clip(seg / peak * 0.707, -1, 1).astype("float32")
                stem = Path(au.get("path") or f"fma_{made:05d}").stem
                sf.write(flac_dir / f"{stem}.flac", seg, SR, format="FLAC", subtype="PCM_16")
                made += 1
                rows.append({
                    "audio_filepath": f"flac/{stem}.flac",
                    "title": (rec.get("title") or "")[:200],
                    "artist": (rec.get("artist") or "")[:200],
                    "album_title": (rec.get("album_title") or "")[:200],
                    "excerpt_s": round(len(seg) / SR, 3),
                    "duration_s": round(len(seg) / SR, 3),
                    # Music is not speech: the correct transcript is nothing at all.
                    "reference_text": "",
                    "vocals": "unknown",      # FMA ships no instrumental/vocal tag; do not guess
                    "license": "CC BY 4.0", "source": "free-music-archive",
                    "source_url": rec.get("url") or f"https://huggingface.co/datasets/{REPO}",
                })
                if made % 100 == 0:
                    print(f"   {made}/{args.tracks}", flush=True)

    if not args.keep_parquet:
        # 11 FMA shards are ~8.8 GB; drop each source file once converted so peak disk
        # stays at roughly one shard plus the growing FLAC output.
        for sp in shard_paths:
            try:
                Path(sp).unlink()
            except OSError:
                pass

    man = args.out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tot = sum(r["duration_s"] for r in rows)
    print(f"\nmusic arm: {len(rows)} excerpts, {tot/3600:.2f} h -> {args.out}")


if __name__ == "__main__":
    main()
