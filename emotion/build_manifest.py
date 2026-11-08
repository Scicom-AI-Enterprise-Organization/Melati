#!/usr/bin/env python3
"""One manifest over every downloaded emotion corpus, labels mapped to Melati's seven. RUNS ON THE BOX.

Rows: {id, corpus, lang, label, speaker, src, transcript, split}. `src` is a file, or a cell of a parquet file
(CAMEO, NonverbalTTS, Expresso keep their audio inside parquet). `split` is "train", or "test" for what no
model may train on: nEMO (Polish, in CAMEO but in no SER training set we know of) and the VoxCeleb half of
NonverbalTTS (the other half is Expresso, whose speakers are in Expresso itself).

Labels: neutral, happy, sad, angry, surprised, fearful, disgusted. Near-synonyms are folded in the usual way
(calm -> neutral as in RAVDESS's own design, excitement/enthusiasm/amused -> happy as IEMOCAP's convention,
anxiety and MER2023's "worried" -> fearful, pleasant surprise -> surprised); anything else (sarcasm, boredom, sleepy, contempt,
apology, assertiveness, concern, encouragement, poker face) is left out rather than forced into a class.

    python emotion/build_manifest.py --raw $MELATI_DATA/emo/raw --out $MELATI_DATA/emo/manifest.jsonl
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

LABELS = ("neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted")
MAP = {"neutral": "neutral", "neutrality": "neutral", "calm": "neutral",
       "happy": "happy", "happiness": "happy", "joy": "happy", "excitement": "happy", "excited": "happy",
       "enthusiasm": "happy", "amused": "happy",
       "sad": "sad", "sadness": "sad",
       "angry": "angry", "anger": "angry",
       "surprise": "surprised", "surprised": "surprised", "ps": "surprised", "pleasant_surprise": "surprised",
       "fear": "fearful", "fearful": "fearful", "anxiety": "fearful", "anxious": "fearful", "worried": "fearful",
       "disgust": "disgusted", "disgusted": "disgusted"}
LANG = {"English": "en", "French": "fr", "Italian": "it", "Spanish": "es", "Polish": "pl", "German": "de",
        "Russian": "ru", "Bengali": "bn"}
HELD_OUT_CORPORA = {"nemo"}


def lab(x):
    return MAP.get(str(x or "").strip().lower().replace(" ", "_"))


def parquet_rows(files, audio_col="audio"):
    import pyarrow.parquet as pq
    for f in files:
        t = pq.read_table(f, columns=[c for c in pq.read_schema(f).names if c != audio_col])
        for i, r in enumerate(t.to_pylist()):
            yield f, i, r


def cameo(raw):
    for f, i, r in parquet_rows(sorted((raw / "cameo").rglob("*.parquet"))):
        corpus = Path(f).name.split("-")[0]
        y = lab(r["emotion"])
        if y:
            yield {"id": f"cameo/{corpus}/{r['file_id']}", "corpus": corpus, "lang": LANG.get(r["language"], r["language"]),
                   "label": y, "speaker": f"{corpus}/{r.get('speaker_id')}", "transcript": r.get("transcription"),
                   "src": {"parquet": str(f), "row": i, "col": "audio"},
                   "split": "test" if corpus in HELD_OUT_CORPORA else "train"}


def nonverbaltts(raw):
    for f, i, r in parquet_rows(sorted((raw / "nonverbaltts").rglob("*.parquet"))):
        y = lab(r.get("Emotion"))
        if not y:
            continue
        vox = (r.get("data_name") or "").lower().startswith("vox")
        yield {"id": f"nvtts/{r['index']}", "corpus": "nvtts_voxceleb" if vox else "nvtts_expresso", "lang": "en",
               "label": y, "speaker": f"nvtts/{r.get('speaker_id')}", "transcript": r.get("Result"),
               "src": {"parquet": str(f), "row": i, "col": "audio"}, "split": "test" if vox else "train"}


def expresso(raw):
    style = {"default": "neutral", "happy": "happy", "sad": "sad"}     # the read styles that are emotions
    for f, i, r in parquet_rows(sorted((raw / "expresso").rglob("*.parquet"))):
        y = style.get(str(r.get("style") or "").lower())
        if y:
            yield {"id": f"expresso/{r.get('id', i)}", "corpus": "expresso", "lang": "en", "label": y,
                   "speaker": f"expresso/{r.get('speaker_id')}", "transcript": r.get("text"),
                   "src": {"parquet": str(f), "row": i, "col": "audio"}, "split": "train"}


def files(raw, name, exts=(".wav", ".mp3", ".flac")):
    return sorted(p for p in (raw / name).rglob("*") if p.suffix.lower() in exts and "_archives" not in p.parts
                  and "__MACOSX" not in p.parts)


def row(corpus, p, y, lang, speaker=None, transcript=None):
    # the id is the path under the corpus folder: several corpora reuse file names across speakers
    rel = "/".join(p.parts[p.parts.index(corpus) + 1:]) if corpus in p.parts else p.name
    return {"id": f"{corpus}/{rel}", "corpus": corpus, "lang": lang, "label": y,
            "speaker": f"{corpus}/{speaker}" if speaker else None, "transcript": transcript,
            "src": {"file": str(p)}, "split": "train"}


def emodb(raw):           # 03a01Fa.wav: speaker 03, text a01, emotion F (German initials)
    code = {"W": "angry", "E": "disgusted", "A": "fearful", "F": "happy", "T": "sad", "N": "neutral"}
    for p in files(raw, "emodb"):
        m = re.match(r"(\d\d)[a-z]\d\d([A-Z])", p.stem)
        if m and code.get(m.group(2)):
            yield row("emodb", p, code[m.group(2)], "de", m.group(1))


def tess(raw):            # OAF_back_angry.wav
    for p in files(raw, "tess"):
        y = lab(p.stem.split("_")[-1])
        if y:
            yield row("tess", p, y, "en", p.stem.split("_")[0])


def shemo(raw):           # F01A01.wav: gender, speaker, emotion letter
    code = {"A": "angry", "H": "happy", "N": "neutral", "S": "sad", "W": "surprised", "F": "fearful"}
    for p in files(raw, "shemo"):
        m = re.match(r"([FM]\d\d)([AHNSWF])\d\d", p.stem)
        if m:
            yield row("shemo", p, code[m.group(2)], "fa", m.group(1))


def emovo(raw):           # dis-f1-b1.wav
    code = {"dis": "disgusted", "gio": "happy", "neu": "neutral", "pau": "fearful", "rab": "angry", "sor": "surprised",
            "tri": "sad"}
    for p in files(raw, "emovo"):
        parts = p.stem.lower().split("-")
        if parts and parts[0] in code:
            yield row("emovo", p, code[parts[0]], "it", parts[1] if len(parts) > 1 else None)


def by_folder(corpus, lang, code=None):
    """Corpora that file each clip under a folder named for its emotion (TurEV-DB, URDU, AESDD, EmoV-DB)."""
    def run(raw):
        for p in files(raw, corpus):
            for part in reversed(p.parts[:-1]):
                key = part.lower().split("_")[-1]
                y = (code or {}).get(key) or lab(key)
                if y:
                    yield row(corpus, p, y, lang)
                    break
    return run


def meld(raw):            # train_sent_emo.csv etc.; audio inside dia{D}_utt{U}.mp4
    import csv
    root = raw / "meld"
    vids = {p.name: p for p in root.rglob("*.mp4")}
    for c in root.rglob("*_sent_emo.csv"):
        split = c.name.split("_")[0]
        for r in csv.DictReader(c.open(encoding="utf-8", errors="replace")):
            y = lab(r.get("Emotion"))
            p = vids.get(f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4")
            if y and p and split in ("train", "dev", "test"):
                yield {"id": f"meld/{split}/{p.name}", "corpus": "meld", "lang": "en", "label": y,
                       "speaker": f"meld/{r.get('Speaker')}", "transcript": r.get("Utterance"),
                       "src": {"file": str(p)}, "split": "train"}


def emovdb(raw):          # <speaker>_<Emotion>/<file>.wav, one folder per archive
    for p in files(raw, "emovdb"):
        spk, _, emo = p.parent.name.partition("_")
        y = lab(emo)
        if y and spk:
            yield row("emovdb", p, y, "en", spk)


def ased(raw):            # 01Neutral/n1-01-01-01-01.wav
    for p in files(raw, "ased"):
        y = lab(re.sub(r"^\d+", "", p.parent.name))
        if y:
            yield row("ased", p, y, "am", p.stem.split("-")[1] if "-" in p.stem else None)


def turevdb(raw):         # Sound Source/<Emotion>/<id>_<speaker>_<word>.wav; the other folders are filtered copies
    for p in files(raw, "turevdb"):
        if "Sound Source" in p.parts:
            y = lab(p.parent.name)
            parts = p.stem.split("_")
            if y:
                yield row("turevdb", p, y, "tr", parts[1] if len(parts) > 2 else None)


def mer2023(raw):         # emotion/fetch_mer2023.py: labels.jsonl beside 16 kHz audio cut from the videos
    for line in (raw / "mer2023" / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        y = lab(r["label"])
        if y:
            yield {"id": f"mer2023/{r['split']}/{r['name']}", "corpus": "mer2023", "lang": "zh", "label": y,
                   "speaker": None, "transcript": None, "src": {"file": r["audio"]}, "split": "train"}


SOURCES = {"cameo": cameo, "nvtts": nonverbaltts, "expresso": expresso, "emodb": emodb, "tess": tess,
           "shemo": shemo, "emovo": emovo, "meld": meld,
           "turevdb": turevdb, "urdu": by_folder("urdu", "ur"),
           "aesdd": by_folder("aesdd", "el", {"a": "angry", "d": "disgusted", "f": "fearful", "h": "happy", "s": "sad"}),
           "emovdb": emovdb, "ased": ased,
           "polish": by_folder("polish", "pl"), "mer2023": mer2023}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, seen = [], set()
    folder = {"nvtts": "nonverbaltts"}
    for name, fn in SOURCES.items():
        n0 = len(rows)
        if not (args.raw / folder.get(name, name) / ".fetched").exists():   # still downloading, or never fetched
            continue
        try:
            for r in fn(args.raw):
                if r["id"] not in seen:
                    seen.add(r["id"]); rows.append(r)
        except FileNotFoundError:
            pass
        print(f"{name:10s} {len(rows) - n0:7d}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = Counter((r["corpus"], r["label"]) for r in rows)
    corpora = sorted({r["corpus"] for r in rows})
    print(f"{len(rows)} clips, {len(corpora)} corpora -> {args.out}")
    for c in corpora:
        print(f"  {c:16s} " + " ".join(f"{l[:4]} {by[(c, l)]:5d}" for l in LABELS))


if __name__ == "__main__":
    main()
