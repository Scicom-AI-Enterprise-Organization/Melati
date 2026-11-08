#!/usr/bin/env python3
"""Build the music / noise arms. Silence is not the only trigger -- background music and
noise produce hallucinated text too, and in production that is the far more common case.

Two arms:

  nonspeech        FSD50K clips containing NO speech at all (music, or environmental /
                   mechanical noise). Reference is the empty string, so any output is a
                   hallucination -- same reading as the `silence` arm, but with energy
                   present, which is the harder and more realistic condition.

  speech_in_noise  genuine Malay speech from the `genuine` arm mixed with that music /
                   noise at a controlled SNR. Reference is the REAL transcript. This is
                   the operational case: an agent talking over background music. It
                   catches the failure the pure arms cannot -- transcription degrading
                   into fabrication as evidence weakens, rather than from nothing at all.

FSD50K is CC BY 4.0, so these are redistributable. Clips carrying any voice label are
excluded, since speech in the "non-speech" arm would invalidate the ground truth.

  python scripts/build_nonspeech_arm.py --music 600 --noise 600
"""
import argparse, csv, io, os, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
REPO = "Fhrozen/FSD50k"
SPLIT = "eval"

# Any of these means a human voice is present -> cannot be in a "non-speech" arm.
VOICE = {
    "Speech", "Human_voice", "Singing", "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking", "Conversation",
    "Chatter", "Shout", "Screaming", "Whispering", "Laughter", "Crying_and_sobbing",
    "Yell", "Male_singing", "Female_singing", "Child_singing", "Speech_synthesizer",
    "Babbling", "Whoop", "Sigh", "Gasp", "Groan", "Grunt", "Burping_and_eructation",
    "Human_group_actions", "Cheering", "Applause", "Clapping", "Booing", "Chant",
}
MUSIC = {"Music", "Musical_instrument"}
SNRS = [20, 10, 5, 0, -5]


def to_mono16k(raw: bytes) -> np.ndarray | None:
    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x if x.size >= SR // 2 else None      # need at least 0.5 s


def norm_peak(x, target=0.707):
    p = float(np.abs(x).max())
    return np.clip(x / p * target, -1, 1).astype("float32") if p > 0 else x


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so the mixture sits at the requested SNR, then peak-normalise."""
    if noise.size < speech.size:                  # tile the noise to cover the speech
        noise = np.tile(noise, int(np.ceil(speech.size / noise.size)))
    noise = noise[: speech.size]
    ps, pn = float(np.mean(speech ** 2)), float(np.mean(noise ** 2))
    if ps <= 0 or pn <= 0:
        return norm_peak(speech)
    scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10))))
    return norm_peak(speech + noise * scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--music", type=int, default=600)
    ap.add_argument("--noise", type=int, default=600)
    ap.add_argument("--mix", type=int, default=240, help="speech clips to mix (x len(SNRS) outputs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="eval", choices=["eval", "dev"],
                    help="FSD50K split: 'eval' is the benchmark, 'dev' is for training")
    ap.add_argument("--out-nonspeech", type=Path, default=ROOT / "audio" / "nonspeech")
    ap.add_argument("--out-mix", type=Path, default=ROOT / "audio" / "speech_in_noise")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    rng = random.Random(args.seed)
    global SPLIT
    SPLIT = args.split

    labels = hf_hub_download(REPO, f"labels/{args.split}.csv", repo_type="dataset", token=tok)
    rows = list(csv.DictReader(open(labels, encoding="utf-8")))
    music_ids, noise_ids = [], []
    for r in rows:
        ls = set(r["labels"].split(","))
        if ls & VOICE:
            continue
        (music_ids if ls & MUSIC else noise_ids).append((r["fname"], r["labels"]))
    rng.shuffle(music_ids); rng.shuffle(noise_ids)
    music_ids, noise_ids = music_ids[: args.music], noise_ids[: args.noise]
    print(f"selected {len(music_ids)} music + {len(noise_ids)} noise (voice-free) from {len(rows)} eval clips")

    flac_dir = args.out_nonspeech / "flac"; flac_dir.mkdir(parents=True, exist_ok=True)
    man = args.out_nonspeech / "manifest.csv"
    ns_rows = list(csv.DictReader(man.open(encoding="utf-8"))) if man.exists() else []
    have = {r["audio_filepath"] for r in ns_rows}
    pool = {"music": [], "noise": []}

    def fetch_one(kind, fname, lab):
        """One clip: reuse if already converted, else download and convert."""
        rel = f"flac/{kind}_{fname}.flac"
        dest = flac_dir / f"{kind}_{fname}.flac"
        if dest.exists():
            try:
                x, _ = sf.read(dest, dtype="float32")
                return kind, rel, fname, lab, x, True
            except Exception:
                dest.unlink(missing_ok=True)
        try:
            p = hf_hub_download(REPO, f"clips/{SPLIT}/{fname}.wav", repo_type="dataset", token=tok)
            x = to_mono16k(Path(p).read_bytes())
        except Exception:
            return None
        if x is None:
            return None
        x = norm_peak(x)
        sf.write(dest, x, SR, format="FLAC", subtype="PCM_16")
        return kind, rel, fname, lab, x, False

    # Serial per-file downloads ran ~12/min. The Hub is fine with concurrency (unlike
    # Wikimedia, which throttles bots), so fan out.
    jobs = [("music", f, l) for f, l in music_ids] + [("noise", f, l) for f, l in noise_ids]
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(fetch_one, *j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if not r:
                continue
            kind, rel, fname, lab, x, cached = r
            pool[kind].append(x)
            if rel not in have:
                have.add(rel)
                ns_rows.append({
                    "audio_filepath": rel, "kind": kind, "fsd50k_id": fname,
                    "fsd50k_labels": lab, "duration_s": round(len(x) / SR, 3),
                    "reference_text": "",        # no speech: any output is a hallucination
                    "license": "CC BY 4.0", "source": "FSD50K",
                    "source_url": f"https://huggingface.co/datasets/{REPO}",
                })
            if done % 200 == 0:
                print(f"   {done}/{len(jobs)}  music={len(pool['music'])} noise={len(pool['noise'])}", flush=True)
    print(f"  fetched: music={len(pool['music'])} noise={len(pool['noise'])}")

    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_filepath", "kind", "fsd50k_id", "fsd50k_labels",
                                          "duration_s", "reference_text", "license", "source", "source_url"])
        w.writeheader(); w.writerows(ns_rows)
    print(f"nonspeech -> {len(ns_rows)} clips")

    # ---- speech_in_noise ----
    gman = ROOT / "audio" / "genuine" / "manifest.csv"
    if not gman.exists():
        print("!! genuine arm missing; skipping speech_in_noise"); return
    grows = [r for r in csv.DictReader(gman.open(encoding="utf-8")) if float(r["duration_s"]) <= 20]
    # Prefer the clips that actually carry a high-risk phrase; they are the ones where a
    # hallucinated `terima kasih` and a real one become genuinely confusable.
    grows.sort(key=lambda r: (r["has_target_phrase"] != "true", rng.random()))
    grows = grows[: args.mix]

    mdir = args.out_mix / "flac"; mdir.mkdir(parents=True, exist_ok=True)
    mrows = []
    for i, r in enumerate(grows):
        sp, _ = sf.read(ROOT / "audio" / "genuine" / r["audio_filepath"], dtype="float32")
        for snr in SNRS:
            kind = "music" if i % 2 == 0 else "noise"
            if not pool[kind]:
                continue
            bg = pool[kind][rng.randrange(len(pool[kind]))]
            y = mix_at_snr(sp, bg, snr)
            stem = f"{Path(r['audio_filepath']).stem}_{kind}_snr{snr:+d}"
            sf.write(mdir / f"{stem}.flac", y, SR, format="FLAC", subtype="PCM_16")
            mrows.append({
                "audio_filepath": f"flac/{stem}.flac", "lang": r["lang"],
                "text": r["text"], "snr_db": snr, "background": kind,
                "duration_s": round(len(y) / SR, 3),
                "has_target_phrase": r["has_target_phrase"],
                "target_phrases": r["target_phrases"],
                "speech_source": r["source"], "license": "CC BY 4.0",
                "note": "genuine speech mixed with voice-free FSD50K background",
            })
        if (i + 1) % 60 == 0:
            print(f"   mixed {i+1}/{len(grows)}", flush=True)

    mm = args.out_mix / "manifest.csv"
    with mm.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mrows[0]))
        w.writeheader(); w.writerows(mrows)
    print(f"speech_in_noise -> {len(mrows)} clips ({len(grows)} speech x {len(SNRS)} SNRs)")


if __name__ == "__main__":
    main()
