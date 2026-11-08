#!/usr/bin/env python3
"""The AMI Meeting Corpus as a non-verbal source: meetings with every word and vocal sound timed. RUNS ON THE BOX.

AMI (Carletta et al. 2005; CC BY 4.0, annotations included): 171 meetings, about 100 hours of English, four people
each wearing a headset. The manual annotations (ami_public_manual_1.6.2.zip) time every word of every speaker
(forced to the hand transcripts) and every vocal sound the transcribers marked: 16.5k laughs, 1.1k coughs, 8.9k
others, and a few dozen sighs, sharp breaths and hums. Breathing in general is not marked, so a sound with no mark
is unknown, not nothing. The audio is the headset mix: one channel per meeting with everyone at close range, which
is what a podcast recording is.

Splits are AMI's full-corpus ASR partition, by meeting: its dev meetings are validation, its eval meetings test.

    python nonverbal/ami.py fetch --root $MELATI_DATA/nv/raw/ami      # the annotations and every headset mix
    python nonverbal/ami.py stats --root $MELATI_DATA/nv/raw/ami
"""
import argparse, os, re, subprocess, sys, zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

BASE = "https://groups.inf.ed.ac.uk/ami"
ANNOTATIONS = "ami_public_manual_1.6.2.zip"
DEV = {f"{s}{p}" for s in ("ES2011", "IS1008", "TS3004") for p in "abcd"} | {"IB4001", "IB4002", "IB4003", "IB4004",
                                                                             "IB4010", "IB4011"}
EVAL = {f"{s}{p}" for s in ("EN2002", "ES2004", "IS1009", "TS3003") for p in "abcd"}
FILLERS = {"um", "uh", "mm", "hmm", "hm", "mm-hmm", "uh-huh", "mhm", "er", "erm", "uh-uh", "mm-mm"}
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def split_of(meeting: str) -> str:
    return "validation" if meeting in DEV else "test" if meeting in EVAL else "train"


def sound_label(kind: str) -> str:
    """A vocal sound's type as a head label (melati.nv_gaps.HEAD_LABELS)."""
    k = kind.lower()
    if k.startswith("laugh"):
        return "laugh"
    if k.startswith("cough"):
        return "cough"
    if k.startswith("sigh"):
        return "sigh"
    if "throat" in k:
        return "throat clearing"
    if "sneez" in k:
        return "sneeze"
    if "sniff" in k:
        return "sniff"
    if any(b in k for b in ("inhale", "exhale", "intake of air", "breath")):
        return "breath"
    if "fill" in k:
        return "filler"
    return "other"


def meetings(z: zipfile.ZipFile):
    return sorted({Path(n).name.split(".")[0] for n in z.namelist()
                   if n.startswith("words/") and n.endswith(".words.xml")})


def read(z: zipfile.ZipFile, meeting: str):
    """-> (words, sounds) over every speaker of one meeting, sorted by start. words: {w, s, e, spk, label} with
    label "filler" for um / uh / mm-hmm and the like, else "speech"; sounds: {kind, s, e, spk, label}. Punctuation
    tokens and anything without times are dropped."""
    words, sounds = [], []
    for n in sorted(n for n in z.namelist() if n.startswith(f"words/{meeting}.") and n.endswith(".words.xml")):
        spk = n.split(".")[1]
        text = z.read(n).decode("iso-8859-1")
        for m in re.finditer(r"<(w|vocalsound)\b([^>]*?)(/>|>([^<]*)</w>)", text):
            a = dict(_ATTR.findall(m.group(2)))
            if "starttime" not in a or "endtime" not in a:
                continue
            s, e = float(a["starttime"]), float(a["endtime"])
            if m.group(1) == "w":
                if a.get("punc") == "true" or e <= s:
                    continue
                w = (m.group(4) or "").strip()
                words.append({"w": w, "s": s, "e": e, "spk": spk,
                              "label": "filler" if w.lower().strip(".,?!") in FILLERS else "speech"})
            elif e > s:
                kind = a.get("type", "other")
                sounds.append({"kind": kind, "s": s, "e": e, "spk": spk, "label": sound_label(kind)})
    return sorted(words, key=lambda w: w["s"]), sorted(sounds, key=lambda v: v["s"])


def union(spans):
    """Overlapping spans merged: where anyone is speaking a word, as {s, e}."""
    out = []
    for w in sorted(spans, key=lambda w: w["s"]):
        if out and w["s"] <= out[-1]["e"]:
            out[-1]["e"] = max(out[-1]["e"], w["e"])
        else:
            out.append({"s": w["s"], "e": w["e"]})
    return out


def wav_path(root: Path, meeting: str) -> Path:
    return root / "audio" / f"{meeting}.Mix-Headset.wav"


def fetch_one(job):
    url, dst = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(".part")
    for _ in range(5):
        r = subprocess.run(["curl", "-sSfL", "--retry", "3", "-C", "-", "--speed-time", "60", "--speed-limit", "10000",
                            "-o", str(part), url], capture_output=True)
        if r.returncode == 0:
            with part.open("rb") as fh:
                if fh.read(4) == b"RIFF":
                    part.rename(dst)
                    return dst.name, None
            part.unlink()
    return dst.name, r.stderr.decode(errors="replace")[-200:]


def fetch(root: Path, workers=8):
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / ANNOTATIONS
    if not zpath.exists():
        subprocess.run(["curl", "-sSfL", "-o", str(zpath), f"{BASE}/AMICorpusAnnotations/{ANNOTATIONS}"], check=True)
    with zipfile.ZipFile(zpath) as z:
        ms = meetings(z)
    jobs = [(f"{BASE}/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav", wav_path(root, m)) for m in ms
            if not wav_path(root, m).exists()]
    print(f"{len(ms)} meetings, {len(jobs)} to fetch", flush=True)
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        for name, err in ex.map(fetch_one, jobs):
            if err:
                failed.append(name)
                print(f"  failed {name}: {err}", flush=True)
    have = sum(wav_path(root, m).exists() for m in ms)
    print(f"FETCH_DONE {have}/{len(ms)} meetings, {len(failed)} failed", flush=True)


def stats(root: Path):
    with zipfile.ZipFile(root / ANNOTATIONS) as z:
        ms = meetings(z)
        c = Counter()
        for m in ms:
            words, sounds = read(z, m)
            c[(split_of(m), "meetings")] += 1
            c[(split_of(m), "words")] += len(words)
            for v in sounds:
                c[(split_of(m), v["label"])] += 1
    for k in sorted(c):
        print(k, c[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "stats"])
    ap.add_argument("--root", type=Path, default=Path(f"{MELATI_DATA}/nv/raw/ami"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fetch(args.root, args.workers) if args.step == "fetch" else stats(args.root)


if __name__ == "__main__":
    sys.exit(main())
