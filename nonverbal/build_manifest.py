#!/usr/bin/env python3
"""Training clips for Melati's non-verbal heads, cut to look like what they will be asked about. RUNS ON THE BOX.

The pipeline asks about a CANDIDATE: sound in a gap between two aligned words, at most 2.5 s, cut out and padded
with 0.3 s of silence on each side (melati.nv_gaps.candidates / clip_of). Every training clip is made the same way:

  PodcastFillers  (Zhu et al. 2022; Adobe licence, non-commercial research) English podcasts with every event timed
                  by hand. Each event is cut by its own times: Uh, Um and Agree ("mhm") -> filler; Words,
                  Repetitions, "like", "you know" -> speech; Breath; Laughter; Music and Noise -> noise. Overlap and
                  Other are left out. The episode split is kept (it is by show), so the test episodes, which
                  scripts/eval_fillers.py scores Melati on, are never trained on.
  VocalSound      (Gong et al. 2022; CC BY-SA 4.0) 21,024 recordings of laughter, sighs, coughs, throat clearing,
                  sneezes and sniffs by 3,365 people. Each is trimmed to its sound (frames above the recording's
                  noise floor + 6 dB and within 40 dB of its peak). Split by speaker, 75/10/15, as its authors split
                  it (their lists are no longer published).

  COUGHVID        (Orlandic et al. 2021; CC BY 4.0) crowdsourced coughs, the ones its authors' own detector is
                  confident about. It stands in for VocalSound's coughs in the head that is released: VocalSound's
                  ShareAlike and PodcastFillers' non-commercial terms cannot both hold for one set of weights.
  Coswara         (Bhattacharya et al. 2023; CC BY 4.0) a heavy and a shallow cough from each of ~2,700 people,
                  recorded on their own phones and rated by the authors' listeners; the good and excellent ones,
                  split by participant. Its rows and COUGHVID's share the pool "coughs", which train.py weighs as
                  one corpus: two sets of phone-recorded coughs must not outweigh every corpus with speech in it.

Clips are 16 kHz PCM16 under --clips/<corpus>/ (default /dev/shm: ~105k files of a second or two, which the
network disk creates at a few a second; they are derived and rebuilt in minutes), one manifest row each in
<out>/manifest.jsonl: id, corpus, split, label, src, dur (the sound's length before padding). VocalSound is read
straight from its release zip. Labels are melati.nv_gaps.HEAD_LABELS.

    python nonverbal/build_manifest.py --podcastfillers $MELATI_DATA/bench/podcastfillers \\
        --vocalsound $MELATI_DATA/nv/raw/vocalsound/vs_release_44k.zip --out $MELATI_DATA/nv
    python nonverbal/build_manifest.py --coswara $MELATI_DATA/nv/raw/coswara --out $MELATI_DATA/nv \\
        --name manifest_coswara
"""
import argparse, csv, json, subprocess, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati.nv_gaps import HEAD_LABELS, PAD_S, SR  # noqa: E402

MAX_S = 2.5                                     # melati.nv_gaps.candidates' max_len
PF_MAP = {"Uh": "filler", "Um": "filler", "Agree": "filler", "Words": "speech", "Repetitions": "speech",
          "Like": "speech", "You know": "speech", "Breath": "breath", "Laughter": "laugh", "Music": "noise",
          "Noise": "noise"}
VS_MAP = {"laughter": "laugh", "sigh": "sigh", "cough": "cough", "throatclearing": "throat clearing",
          "sneeze": "sneeze", "sniff": "sniff"}


def padded(x):
    pad = np.zeros(int(PAD_S * SR), np.float32)
    return np.concatenate([pad, x[: int(MAX_S * SR)].astype(np.float32), pad])


def write(path: Path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1), SR, subtype="PCM_16")


def resample(x, sr):
    if sr == SR:
        return x
    import soxr
    return soxr.resample(x, sr, SR)


def pf_episode(job):
    """Every kept event of one episode, cut from the episode once."""
    wav, events, out = job
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = resample(x.mean(1), sr)
    rows = []
    for e in events:
        a, b = int(float(e["event_start_inepisode"]) * SR), int(float(e["event_end_inepisode"]) * SR)
        if b - a < int(0.03 * SR) or a >= len(x):
            continue
        dst = out / e["episode_split_subset"] / f"{e['pfID']}.wav"
        if not dst.exists():
            write(dst, padded(x[a:b]))
        rows.append({"id": f"pf/{e['pfID']}", "corpus": "podcastfillers", "split": e["episode_split_subset"],
                     "label": PF_MAP[e["label_full_vocab"]], "orig": e["label_full_vocab"],
                     "src": {"file": str(dst)}, "dur": round((b - a) / SR, 3),
                     "episode": e["podcast_filename"]})
    return rows


def trim(x, floor_db=6.0, rel_db=-40.0, frame=0.01):
    n = int(frame * SR)
    m = len(x) // n
    if m < 3:
        return x
    db = 10 * np.log10(np.mean(x[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    thr = max(np.percentile(db, 10) + floor_db, db.max() + rel_db)
    idx = np.flatnonzero(db >= thr)
    return x[idx[0] * n:(idx[-1] + 1) * n] if len(idx) else x


_ZIPS = {}


def vs_clip(job):
    zpath, member, label, split, out = job
    import io, zipfile
    z = _ZIPS.get(zpath) or _ZIPS.setdefault(zpath, zipfile.ZipFile(zpath))      # one handle per worker
    x, sr = sf.read(io.BytesIO(z.read(member)), dtype="float32", always_2d=True)
    x = trim(resample(x.mean(1), sr))
    stem = Path(member).stem
    dst = out / split / f"{stem}.wav"
    if not dst.exists():
        write(dst, padded(x))
    return {"id": f"vs/{stem}", "corpus": "vocalsound", "split": split, "label": label,
            "src": {"file": str(dst)}, "dur": round(min(len(x) / SR, MAX_S), 3)}


def podcastfillers(root: Path, out: Path):
    csv_path = next(root.rglob("PodcastFillers.csv"))
    wav_root = next(root.rglob("episode_wav"))
    by_ep = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["label_full_vocab"] in PF_MAP:
                by_ep[(r["episode_split_subset"], r["podcast_filename"])].append(r)
    jobs = []
    for (split, ep), evs in sorted(by_ep.items()):
        wav = wav_root / split / f"{ep}.wav"
        if wav.exists():
            jobs.append((str(wav), evs, out))
        else:
            print(f"  [podcastfillers] no audio for {split}/{ep}", flush=True)
    return jobs


def speaker_split(speaker: str) -> str:
    """Speaker-disjoint, deterministic: 75% train, 10% validation, 15% test by a hash of the speaker id."""
    import hashlib
    b = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 100
    return "train" if b < 75 else "validation" if b < 85 else "test"


def vocalsound(zpath: Path, out: Path):
    """Labels from the file names (f0003_0_cough.wav: sex, speaker, session, label). The release's own split lists
    are in neither its audio zip nor its repository any more, so the split is by speaker as the authors made theirs
    (speaker_split): nobody's voice is in two splits."""
    import zipfile
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n
                 and not Path(n).name.startswith("._")]
    jobs = []
    for n in sorted(names):
        stem = Path(n).stem
        label = VS_MAP.get(stem.split("_")[-1])
        if label:
            jobs.append((str(zpath), n, label, speaker_split(stem.split("_")[0]), out))
    print(f"  [vocalsound] {len(names)} wavs, {len(jobs)} kept, "
          f"{len({Path(j[1]).stem.split('_')[0] for j in jobs})} speakers", flush=True)
    return jobs


def ffmpeg_bytes(raw: bytes):
    """16 kHz mono float32 from any container ffmpeg reads (COUGHVID mixes webm, ogg and wav)."""
    import subprocess
    r = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR),
                        "pipe:1"], input=raw, stdout=subprocess.PIPE)
    return np.frombuffer(r.stdout, np.float32).copy() if r.returncode == 0 else None


def cv_clip(job):
    zpath, member, uuid, split, out = job
    import zipfile
    z = _ZIPS.get(zpath) or _ZIPS.setdefault(zpath, zipfile.ZipFile(zpath))
    x = ffmpeg_bytes(z.read(member))
    if x is None or len(x) < SR // 5:
        return None
    x = trim(x)
    dst = out / split / f"{uuid}.wav"
    if not dst.exists():
        write(dst, padded(x))
    return {"id": f"cv/{uuid}", "corpus": "coughvid", "pool": "coughs", "split": split, "label": "cough",
            "src": {"file": str(dst)}, "dur": round(min(len(x) / SR, MAX_S), 3)}


def coughvid(zpath: Path, out: Path, n=4000, min_p=0.9, seed=0):
    """COUGHVID (Orlandic et al. 2021; CC BY 4.0): crowdsourced cough recordings, each scored by the authors' own
    cough detector (`cough_detected`); only confident ones, at most n, split by recording as VocalSound is."""
    import csv, io, random, zipfile
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        meta = next(m for m in names if m.endswith("metadata_compiled.csv"))
        rows = list(csv.DictReader(io.StringIO(z.read(meta).decode("utf-8"))))
    audio = {Path(m).stem: m for m in names if m.lower().endswith((".webm", ".ogg", ".wav"))}
    keep = [r["uuid"] for r in rows if r.get("cough_detected") and float(r["cough_detected"]) >= min_p
            and r["uuid"] in audio]
    random.Random(seed).shuffle(keep)
    keep = sorted(keep[:n])
    print(f"  [coughvid] {len(rows)} recordings, {len(keep)} kept (cough_detected >= {min_p})", flush=True)
    return [(str(zpath), audio[u], u, speaker_split(u), out) for u in keep]


def coswara_date(job):
    """One collection date's archive (its parts concatenated, read as a stream): the kept coughs, trimmed and padded."""
    import io, tarfile
    parts, keep, out = job
    cat = subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE)
    rows = []
    with tarfile.open(fileobj=cat.stdout, mode="r|gz") as tar:
        for m in tar:
            p = Path(m.name)
            if not m.isfile() or f"{p.parent.name}_{p.stem}" not in keep:
                continue
            try:
                x, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32", always_2d=True)
            except Exception:                     # a handful of uploads are empty or truncated
                continue
            x = resample(x.mean(1), sr)
            if len(x) < SR // 5:
                continue
            x = trim(x)
            pid, kind = p.parent.name, p.stem
            split = speaker_split(pid)
            dst = out / split / f"{pid}_{kind}.wav"
            if not dst.exists():
                write(dst, padded(x))
            rows.append({"id": f"cw/{pid}_{kind}", "corpus": "coswara", "pool": "coughs", "split": split,
                         "label": "cough", "src": {"file": str(dst)}, "dur": round(min(len(x) / SR, MAX_S), 3)})
    cat.wait()
    return rows


def coswara(root: Path, out: Path, min_quality=1):
    """Coswara (Bhattacharya et al. 2023; CC BY 4.0): each participant's heavy and shallow cough where the authors'
    listeners rated it good or excellent (annotations/cough-*_labels.csv: 2 excellent, 1 good, 0 bad). One job per
    collection date; the participant decides the split, as the speaker does for VocalSound."""
    keep = set()
    for kind in ("cough-heavy", "cough-shallow"):
        with (root / "annotations" / f"{kind}_labels.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.reader(fh):
                if len(r) >= 2 and r[1].strip().isdigit() and int(r[1]) >= min_quality:
                    keep.add(r[0].strip())
    jobs = [(sorted(str(p) for p in d.glob(f"{d.name}.tar.gz.*")), keep, out)
            for d in sorted(root.iterdir()) if d.is_dir() and d.name.isdigit()]
    jobs = [j for j in jobs if j[0]]
    print(f"  [coswara] {len(keep)} coughs rated good or excellent, {len(jobs)} archives", flush=True)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--podcastfillers", type=Path, default=None)
    ap.add_argument("--vocalsound", type=Path, default=None, help="the release zip (vs_release_44k.zip)")
    ap.add_argument("--coughvid", type=Path, default=None, help="COUGHVID's public_dataset_v3.zip")
    ap.add_argument("--coswara", type=Path, default=None, help="a clone of iiscleap/Coswara-Data")
    ap.add_argument("--out", type=Path, required=True, help="manifest and stats")
    ap.add_argument("--name", default="manifest", help="manifest file name (<out>/<name>.jsonl)")
    ap.add_argument("--clips", type=Path, default=Path("/dev/shm/melati_nv/clips"))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        if args.podcastfillers:
            for rs in ex.map(pf_episode, podcastfillers(args.podcastfillers, args.clips / "podcastfillers")):
                rows += rs
            print(f"[podcastfillers] {len(rows)} clips", flush=True)
        if args.vocalsound:
            n = len(rows)
            rows += list(ex.map(vs_clip, vocalsound(args.vocalsound, args.clips / "vocalsound"), chunksize=64))
            print(f"[vocalsound] {len(rows) - n} clips", flush=True)
        if args.coughvid:
            n = len(rows)
            rows += [r for r in ex.map(cv_clip, coughvid(args.coughvid, args.clips / "coughvid"), chunksize=32) if r]
            print(f"[coughvid] {len(rows) - n} clips", flush=True)
        if args.coswara:
            n = len(rows)
            for rs in ex.map(coswara_date, coswara(args.coswara, args.clips / "coswara")):
                rows += rs
            print(f"[coswara] {len(rows) - n} clips", flush=True)
    assert all(r["label"] in HEAD_LABELS for r in rows)
    man = args.out / f"{args.name}.jsonl"
    with man.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = defaultdict(Counter)
    for r in rows:
        stats[f"{r['corpus']}/{r['split']}"][r["label"]] += 1
    summary = {k: dict(v.most_common()) for k, v in sorted(stats.items())}
    (args.out / f"{args.name}_stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"MANIFEST_DONE {len(rows)} -> {man}")


if __name__ == "__main__":
    main()
