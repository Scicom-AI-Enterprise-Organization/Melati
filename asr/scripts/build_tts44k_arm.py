#!/usr/bin/env python3
"""A replay arm for the languages Common Voice 22 does not give us, from Sidon's labeled44k. RUNS ON THE BOX.

`Scicom-intl/TTS-Clean44k`, config `labeled44k`: 2.87 M clips at 44.1-48 kHz with text. Its `lang`
column is the language of the TEXT, and some sources are translation pairs (Kabyle or Tamazight
audio with English or Arabic text, Nepali speech with English text): training on those teaches
Whisper to translate. So only named SAME-LANGUAGE transcription sources are taken (`SOURCES`),
each checked by name, never by its `lang` label alone. Kannada and Gujarati are the reason the arm
exists (the CV22 pool has none of either); the other Indic sources are there because the first
fine-tunes drifted those languages to Tamil script.

A named source is still not enough: IndicTTS has English sentences read by the same speakers, under
the same source and `lang` (an IndicTTS-Bengali row reads "The Gabriel voice of the Samurai rang out"),
and training those under <|bn|> would teach Whisper to write English when told Bengali. So a clip is
kept only when at least --min-script of its letters are in its language's own script (`SCRIPT`).

Clips are drawn per source (up to --per-lang per language, spread over speakers where the source
names them), decoded, resampled to 16 kHz mono FLAC, and listed in a manifest in the CV22 arm's
format (asr/train/build_mix.py reads it with --tts44k).

    python asr/scripts/build_tts44k_arm.py --out $MELATI_DATA/asr/audio_train/tts44k --per-lang 600
"""
import argparse, csv, io, json, os, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO, CONFIG = "Scicom-intl/TTS-Clean44k", "labeled44k"
# source -> Whisper language code; each is a transcription of its own audio (checked by name and card)
SOURCES = {"IndicTTS_Kannada": "kn", "gujarati-f-openslr": "gu", "IndicTTS_Tamil": "ta", "openslr65-tamil": "ta",
           "IndicTTS_Malayalam": "ml", "IndicTTS_Marathi": "mr", "IndicTTS-Bengali": "bn", "IndicTTS-Hindi": "hi"}
SR = 16000
# Unicode block of each language's script: a clip's text must be written in it
SCRIPT = {"bn": (0x0980, 0x09FF), "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F), "kn": (0x0C80, 0x0CFF),
          "gu": (0x0A80, 0x0AFF), "ml": (0x0D00, 0x0D7F), "ta": (0x0B80, 0x0BFF)}


def script_share(text, lang):
    """Share of the text's letters (its script's characters or Latin letters) that are in `lang`'s script."""
    lo, hi = SCRIPT[lang]
    own = sum(lo <= ord(ch) <= hi for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    return own / (own + latin) if own + latin else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-lang", type=int, default=600)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-script", type=float, default=0.8, help="share of letters in the language's own script")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    import soxr
    from huggingface_hub import HfApi, HfFileSystem
    tok = os.environ.get("HF_TOKEN")
    files = sorted(f for f in HfApi(token=tok).list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(CONFIG + "/") and f.endswith(".parquet"))
    fs = HfFileSystem(token=tok)

    def index(f):
        """-> [(file, row group, row in group, source, speaker)] for wanted sources whose text is written in
        their language's script; reads three small columns, never the audio."""
        pf = pq.ParquetFile(fs.open(f"datasets/{REPO}/{f}"))
        hits, wrong = [], 0
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["source", "speaker", "text"]).to_pydict()
            for i, s in enumerate(t["source"]):
                if s not in SOURCES:
                    continue
                if script_share(t["text"][i] or "", SOURCES[s]) >= args.min_script:
                    hits.append((f, g, i, s, t["speaker"][i]))
                else:
                    wrong += 1
        return hits, wrong

    print(f"indexing {len(files)} files", flush=True)
    res = list(ThreadPoolExecutor(args.threads).map(index, files))
    hits = [h for hs, _ in res for h in hs]
    print(f"dropped {sum(w for _, w in res)} rows whose text is not in their language's script", flush=True)
    by_lang = defaultdict(list)
    for h in hits:
        by_lang[SOURCES[h[3]]].append(h)
    print({k: len(v) for k, v in by_lang.items()}, flush=True)

    rng = random.Random(args.seed)
    pick = []
    for lang, hs in by_lang.items():
        by_spk = defaultdict(list)
        for h in hs:
            by_spk[(h[3], h[4])].append(h)
        for v in by_spk.values():
            rng.shuffle(v)
        queues = list(by_spk.values())
        rng.shuffle(queues)
        chosen = []
        while len(chosen) < args.per_lang and any(queues):      # round-robin over speakers
            for q in queues:
                if q and len(chosen) < args.per_lang:
                    chosen.append(q.pop())
        pick += chosen
    by_group = defaultdict(list)
    for h in pick:
        by_group[(h[0], h[1])].append(h)
    (args.out / "flac").mkdir(parents=True, exist_ok=True)

    def fetch(key):
        f, g = key
        pf = pq.ParquetFile(fs.open(f"datasets/{REPO}/{f}"))
        t = pf.read_row_group(g, columns=["audio", "text", "source", "speaker", "duration"]).to_pydict()
        rows = []
        for _, _, i, s, spk in by_group[key]:
            a = t["audio"][i]
            data = a.get("bytes") if isinstance(a, dict) else a
            try:
                x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            except Exception:
                continue
            x = soxr.resample(x.mean(1), sr, SR).astype(np.float32)
            name = f"{SOURCES[s]}_{Path(f).stem}_{g}_{i}.flac"
            sf.write(args.out / "flac" / name, x, SR, format="FLAC")
            rows.append({"audio_filepath": f"flac/{name}", "text": (t["text"][i] or "").strip(), "lang": SOURCES[s],
                         "source": s, "speaker": spk, "duration_s": round(len(x) / SR, 3)})
        return rows

    rows = [r for rs in ThreadPoolExecutor(args.threads).map(fetch, list(by_group)) for r in rs if r["text"]]
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["audio_filepath", "text", "lang", "source", "speaker", "duration_s"])
        w.writeheader()
        w.writerows(rows)
    langs = defaultdict(lambda: [0, 0.0])
    for r in rows:
        langs[r["lang"]][0] += 1
        langs[r["lang"]][1] += r["duration_s"] / 3600
    summary = {k: {"clips": n, "hours": round(h, 2)} for k, (n, h) in sorted(langs.items())}
    (args.out / "languages.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(rows)} clips -> {args.out}/manifest.csv", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
