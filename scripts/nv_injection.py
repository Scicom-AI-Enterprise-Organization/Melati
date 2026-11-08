#!/usr/bin/env python3
"""Where do separators put a laugh, and what does a quality filter think of it? RUNS ON THE BOX.

A controlled test of two things Emilia-Pipe does to non-verbal sounds before anyone can tag them:

  1. separation. Emilia keeps UVR-MDX-Net's "vocals" stem; Melati keeps DBS-TIGER's "speech"
     stem and reads non-verbal sounds from speech + effects. A separator that files a cough under
     "instrumental" or "effects" deletes it from the dataset.
  2. quality filtering. Emilia drops every segment with DNSMOS OVRL < 3.0. If a laugh lowers
     DNSMOS, the filter deletes expressive segments for being expressive.

Stimuli: a real vocalisation from FSD50K (CC BY 4.0; labels Laughter, Giggle, Chuckle, Cough,
Sneeze, Sigh, Gasp, Breathing, Crying, Screaming, Burping, Yawn) placed in the pause between two
Common Voice 22 test utterances of one speaker:  [speech A] 0.3 s [vocalisation] 0.3 s [speech B].
Each stimulus is made twice, with the vocalisation and without it (the pause filled with the
same length of silence), and optionally under a music bed (FMA, CC BY 4.0) at -15 dB.

Measured per stimulus:
  * the energy of the vocalisation window in each separator stem, as a share of the window's
    energy in the mixture (music is subtracted by running the same separator on the
    no-vocalisation twin)
  * DNSMOS OVRL and SOTA-MOS of the whole stimulus, with and without the vocalisation

    python scripts/nv_injection.py --n-per-class 60 --out $MELATI_DATA/exp/nv_injection \\
        --results results/separation/nv_injection.json --endpoints $MELATI_DATA/serve_eval/endpoints.json
"""
import argparse, asyncio, csv, io, json, os, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

REPO_FSD = "Fhrozen/FSD50k"
CLASSES = {  # FSD50K label -> Melati tag; a clip must carry exactly one of these
    "Laughter": "laughs", "Giggle": "laughs", "Chuckle_and_chortle": "laughs", "Cough": "coughs",
    "Sneeze": "sneezes", "Sigh": "sighs", "Gasp": "gasps", "Breathing": "breathes",
    "Crying_and_sobbing": "cries", "Screaming": "screams", "Burping_and_eructation": "burps", "Yawn": "yawns",
}
SPEECH_LABELS = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
                 "Child_speech_and_kid_speaking", "Conversation", "Singing", "Music"}
SR = 44100


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def load16(raw, sr_out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from melati import audio as A
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return A.resample(x.mean(1), sr, sr_out)


async def measure(args, stimuli):
    """Every model through the servers: DBS (front /separate), UVR (uvr /uvr), DNSMOS (tag /dnsmos),
    SOTA-MOS (mos /predict)."""
    from melati import audio as A
    from melati.client import Client, Endpoints
    sem = asyncio.Semaphore(16)
    async with Client(Endpoints(args.endpoints)) as c:
        async def mos(x):
            x24, x16 = A.resample(x, SR, 24000), A.resample(x, SR, 16000)
            d, s_ = await asyncio.gather(c.dnsmos([x16]), c.mos(x24))
            return float(d[0]["OVRL"]), float(s_)

        async def one(st):
            async with sem:
                a, b = st["win"]
                res = {k: st[k] for k in ("id", "tag", "bed", "fsd50k")}
                mix_e = rms(st["x1"][a:b] - st["x0"][a:b]) ** 2          # the vocalisation alone
                d1, d0, v1, v0, m1, m0 = await asyncio.gather(
                    c.separate(st["x1"]), c.separate(st["x0"]), c.uvr(st["x1"]), c.uvr(st["x0"]),
                    mos(st["x1"]), mos(st["x0"]))
                for i, stem in enumerate(("music", "sfx", "speech")):
                    res[f"dbs_{stem}"] = round(rms(d1[i][a:b] - d0[i][a:b]) ** 2 / max(mix_e, 1e-12), 4)
                res["uvr_vocals"] = round(rms(v1[a:b] - v0[a:b]) ** 2 / max(mix_e, 1e-12), 4)
                (res["dnsmos_with"], res["sotamos_with"]), (res["dnsmos_without"], res["sotamos_without"]) = m1, m0
                return res
        rows = []
        for k, r in enumerate(asyncio.as_completed([one(st) for st in stimuli]), 1):
            rows.append(await r)
            if k % 50 == 0:
                print(f"{k}/{len(stimuli)}", flush=True)
        return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--cv22-test", type=Path, default=Path(f"{MELATI_DATA}/asr/cv22/test/manifest.csv"))
    ap.add_argument("--music", type=Path, default=Path(f"{MELATI_DATA}/asr/audio_train/music/manifest.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--endpoints", type=Path, required=True, help="front, uvr, tag and mos services")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from huggingface_hub import hf_hub_download
    from melati import audio as A

    rng = random.Random(args.seed)
    tok = os.environ.get("HF_TOKEN")
    args.out.mkdir(parents=True, exist_ok=True)

    # vocalisation clips: single-class, no speech or music label
    labels = list(csv.DictReader(open(hf_hub_download(REPO_FSD, "labels/eval.csv", repo_type="dataset", token=tok))))
    by_tag = defaultdict(list)
    for r in labels:
        ls = set(r["labels"].split(","))
        hit = [CLASSES[l] for l in ls if l in CLASSES]
        if len(set(hit)) == 1 and not (ls & SPEECH_LABELS):
            by_tag[hit[0]].append(r["fname"])
    picks = []
    for tag, names in sorted(by_tag.items()):
        rng.shuffle(names)
        picks += [(tag, n) for n in names[: args.n_per_class]]
    print({t: min(len(v), args.n_per_class) for t, v in sorted(by_tag.items())}, flush=True)

    speech = defaultdict(list)
    for r in csv.DictReader(args.cv22_test.open(encoding="utf-8")):
        speech[r["client_id"]].append(args.cv22_test.parent / r["audio_filepath"])
    pairs = [v[:2] for v in speech.values() if len(v) >= 2]
    music = [args.music.parent / r["audio_filepath"] for r in csv.DictReader(args.music.open(encoding="utf-8"))]

    stimuli = []
    for k, (tag, fname) in enumerate(picks):
        try:
            nv = load16(Path(hf_hub_download(REPO_FSD, f"clips/eval/{fname}.wav", repo_type="dataset",
                                             token=tok)).read_bytes(), SR)
        except Exception as e:  # noqa: BLE001
            print(f"skip {fname}: {e}", flush=True)
            continue
        nv = nv[: int(4.0 * SR)]
        if rms(nv) == 0:
            continue
        a_path, b_path = rng.choice(pairs)
        a = A.decode(a_path, sr=SR)[0]
        b = A.decode(b_path, sr=SR)[0]
        # the vocalisation at the speech's own level (speech RMS), within +-6 dB
        nv = nv * (rms(np.concatenate([a, b])) / rms(nv)) * 10 ** (rng.uniform(-6, 6) / 20)
        gap = np.zeros(int(0.3 * SR), np.float32)
        with_nv = np.concatenate([a, gap, nv, gap, b]).astype(np.float32)
        without = np.concatenate([a, gap, np.zeros_like(nv), gap, b]).astype(np.float32)
        win = (len(a) + len(gap), len(a) + len(gap) + len(nv))
        for bed in ("none", "music"):
            x1, x0 = with_nv.copy(), without.copy()
            if bed == "music" and music:
                m = A.decode(rng.choice(music), sr=SR)[0]
                m = np.resize(m, len(x1)) * (rms(np.concatenate([a, b])) / max(rms(m), 1e-9)) * 10 ** (-15 / 20)
                x1, x0 = x1 + m, x0 + m
            peak = max(np.abs(x1).max(), np.abs(x0).max(), 1e-9) / 0.89
            stimuli.append({"id": f"{k:04d}_{bed}", "tag": tag, "bed": bed, "fsd50k": fname,
                            "x1": (x1 / peak).astype(np.float32), "x0": (x0 / peak).astype(np.float32), "win": win})
    print(f"{len(stimuli)} stimuli", flush=True)

    rows = asyncio.run(measure(args, stimuli))

    (args.out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    agg = {}
    for tag in sorted({r["tag"] for r in rows}):
        for bed in ("none", "music"):
            rs = [r for r in rows if r["tag"] == tag and r["bed"] == bed]
            if not rs:
                continue
            m = lambda k: round(float(np.median([r[k] for r in rs])), 4)
            agg[f"{tag}/{bed}"] = {
                "n": len(rs), "dbs_speech": m("dbs_speech"), "dbs_sfx": m("dbs_sfx"), "dbs_music": m("dbs_music"),
                "uvr_vocals": m("uvr_vocals"),
                "dbs_speech_lt_half": round(float(np.mean([r["dbs_speech"] < 0.5 for r in rs])), 4),
                "dbs_vocal_lt_half": round(float(np.mean([r["dbs_speech"] + r["dbs_sfx"] < 0.5 for r in rs])), 4),
                "uvr_lt_half": round(float(np.mean([r["uvr_vocals"] < 0.5 for r in rs])), 4),
                "dnsmos_delta": m("dnsmos_with") - m("dnsmos_without"),
                "sotamos_delta": round(float(np.median([r["sotamos_with"] - r["sotamos_without"] for r in rs])), 4),
                "dnsmos_cross_3": round(float(np.mean([r["dnsmos_without"] >= 3.0 > r["dnsmos_with"] for r in rs])), 4),
            }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
