#!/usr/bin/env python3
"""Mine real conversational Malaysian speech from Malaysian-Emilia. RUNS ON THE BOX.

Why this source: the benchmark's positive arm is 52% synthetic TTS and 23% read Wikipedia,
with only 19 genuine `terima kasih` in total. Emilia's Malaysian podcast subset is 671 h of
actual conversational speech carrying 1,392 `terima kasih`, the register that was missing,
and untouched by the benchmark, so it is training-safe by construction.

Licence: CC BY-NC 4.0. Fetched by script, never re-hosted, so the ablation stays
reproducible while the published dataset stays permissively licensed.

The audio ships as a split zip (.z01 + .zip). `zip` is unavailable on the box, so the parts
are concatenated and read with Python's zipfile: for a spanned archive the central
directory offsets are relative to the whole stream, so this is valid.

  python scripts/mine_emilia.py --target-phrase-clips 4000 --general-clips 8000
"""
import argparse, csv, io, os, random, re, unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from scipy.signal import resample_poly

MELATI_ROOT = os.environ.get("MELATI_ROOT", "/root/melati")        # environments, checkouts: deploy/env.sh

REPO = "mesolitica/Malaysian-Emilia-annotated"
PARQUET = "malaysian-emilia-podcast.parquet"
PARTS = ["malaysian-podcast_processed_24k.z01", "malaysian-podcast_processed_24k.zip"]
SR = 16000
_P = re.compile(r"[.?!,;:\"'`´()\[\]…]+")


def norm(t):
    t = unicodedata.normalize("NFC", t or "").strip().casefold()
    return re.sub(r"\s+", " ", _P.sub(" ", t)).strip()


def to16k(raw: bytes):
    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    if x.size < SR // 2:
        return None
    peak = float(np.abs(x).max())
    return np.clip(x / peak * 0.707, -1, 1).astype("float32") if peak > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("audio_train/emilia"))
    ap.add_argument("--raw", type=Path, default=Path("emilia_raw"))
    ap.add_argument("--target-phrase-clips", type=int, default=4000)
    ap.add_argument("--general-clips", type=int, default=8000)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    rng = random.Random(args.seed)
    args.raw.mkdir(parents=True, exist_ok=True)

    # --- annotations ---
    pqf = hf_hub_download(REPO, PARQUET, repo_type="dataset", token=tok)
    t = pq.read_table(pqf, columns=["transcription", "audio_filename", "speech_duration", "gender", "country"])
    ann = t.to_pylist()
    print(f"annotations: {len(ann)} utterances")

    targets = set()
    tf = Path("phrases/targets.csv")
    if tf.exists():
        for r in csv.DictReader(tf.open(encoding="utf-8")):
            if r["lang"] in ("ms", "id", "en"):
                targets.add(norm(r["phrase"]))
    targets |= {norm(x) for x in ["terima kasih", "sama-sama", "selamat", "maaf", "tolong", "sila", "thank you"]}

    hit, general = [], []
    for a in ann:
        d = float(a.get("speech_duration") or 0)
        if not (0.5 <= d <= args.max_duration):
            continue
        nt = norm(a.get("transcription"))
        if not nt:
            continue
        m = sorted({p for p in targets if p and p in nt})
        (hit if m else general).append((a, m))
    rng.shuffle(hit); rng.shuffle(general)
    # Roughly 40% of archive members point at data the published split parts do not contain,
    # so oversample candidates heavily and stop once the targets are actually met rather
    # than assuming every selected utterance is retrievable.
    cand = [(a, m, True) for a, m in hit[: args.target_phrase_clips * 6]] + \
           [(a, m, False) for a, m in general[: args.general_clips * 6]]
    print(f"candidates: {sum(1 for c in cand if c[2])} phrase-carrying + "
          f"{sum(1 for c in cand if not c[2])} general "
          f"(targets {args.target_phrase_clips}/{args.general_clips})")

    # --- audio archive ---
    # Melati: the parts are a true multi-disk zip (`zip -s`), whose central directory addresses
    # (disk, offset) pairs, so concatenating them does NOT make a zip Python can read ("zipfiles
    # that span multiple disks are not supported"). 7-Zip reads the split set as it is: the parts
    # are linked side by side under their own names and extracted once.
    extracted = args.raw / "extracted"
    done = args.raw / ".extracted"
    if not done.exists():
        parts = args.raw / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        for part in PARTS:
            lp = Path(hf_hub_download(REPO, part, repo_type="dataset", token=tok))
            link = parts / part
            if not link.exists():
                link.symlink_to(lp.resolve())
        import subprocess
        seven = os.environ.get("SEVENZIP", f"{MELATI_ROOT}/bin/7zz")
        subprocess.run([seven, "x", str(parts / PARTS[-1]), f"-o{extracted}", "-y", "-bso0", "-bsp0"], check=True)
        done.touch()
    by_base = {}
    for p in extracted.rglob("*"):
        if p.is_file():
            by_base.setdefault(p.name, p)
    print(f"archive members: {len(by_base)}")

    flac = args.out / "flac"; flac.mkdir(parents=True, exist_ok=True)
    rows, miss, got_hit, got_gen = [], 0, 0, 0
    for a, m, is_hit in cand:
        if is_hit and got_hit >= args.target_phrase_clips:
            continue
        if not is_hit and got_gen >= args.general_clips:
            continue
        if got_hit >= args.target_phrase_clips and got_gen >= args.general_clips:
            break
        base = Path(a["audio_filename"]).name
        member = by_base.get(base)
        if member is None:
            miss += 1
            continue
        try:
            x = to16k(member.read_bytes())
        except Exception:
            miss += 1
            continue
        if x is None:
            miss += 1
            continue
        got_hit += is_hit; got_gen += (not is_hit)
        stem = Path(base).stem
        sf.write(flac / f"{stem}.flac", x, SR, format="FLAC", subtype="PCM_16")
        rows.append({
            "audio_filepath": f"flac/{stem}.flac",
            "lang": "ms", "source": "malaysian_emilia_podcast",
            "text": (a.get("transcription") or "").strip()[:2000],
            "duration_s": round(len(x) / SR, 3),
            "license": "CC BY-NC 4.0",
            "has_target_phrase": str(bool(m)).lower(),
            "target_phrases": "|".join(m),
            "gender": a.get("gender", ""), "note": "real conversational Malaysian podcast",
            "author": REPO, "source_url": f"https://huggingface.co/datasets/{REPO}",
        })
        if len(rows) % 500 == 0:
            print(f"   {len(rows)} cut", flush=True)

    man = args.out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    h = sum(r["duration_s"] for r in rows) / 3600
    print(f"\nemilia: {len(rows)} clips, {h:.2f} h  (unretrievable members skipped: {miss})")
    print(f"  with a target phrase: {sum(r['has_target_phrase']=='true' for r in rows)}")


if __name__ == "__main__":
    main()
