#!/usr/bin/env python3
"""Fetch MER2023 (Mandarin film and TV, 5,030 labelled clips) for the emotion heads. RUNS ON THE BOX.

Gated on Hugging Face (MERChallenge/MER2023): academic, non-commercial use, and neither the data nor files derived
from it (label files included) may be handed to third parties. So nothing from it is committed here: the archive
passwords and checksums are read at run time from README_AFTER_APPROVAL.md, which only an approved account can fetch,
and the clips and labels stay under --root on the box.

The training archive is ~140 GB because it also carries MER-SEMI's 73k unlabelled videos; only the labelled ones
are extracted (Train&Val 3,373; MER-MULTI 411; MER-NOISE 412; MER-SEMI's 834 evaluation clips), each as 16 kHz mono.

    python emotion/fetch_mer2023.py --root $MELATI_DATA/emo/raw/mer2023 download   # resumable, then sha256
    python emotion/fetch_mer2023.py --root $MELATI_DATA/emo/raw/mer2023 list       # archive contents
    python emotion/fetch_mer2023.py --root $MELATI_DATA/emo/raw/mer2023 extract
"""
import argparse, csv, hashlib, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MELATI_ROOT = os.environ.get("MELATI_ROOT", "/root/melati")        # environments, checkouts: deploy/env.sh

REPO = "MERChallenge/MER2023"
TRAIN_PARTS = [f"mer2023train.z0{i}" for i in range(1, 7)] + ["mer2023train.zip"]
SMALL = ["mer2023test1&2.zip", "test-labels.zip"]
SEVEN = f"{MELATI_ROOT}/bin/7zz"
VIDEO = (".mp4", ".avi", ".mkv", ".flv", ".mov", ".webm")


def readme(root: Path) -> str:
    p = root / "README_AFTER_APPROVAL.md"
    if not p.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, p.name, repo_type="dataset", local_dir=str(root))      # small: the hub client is fine
    return p.read_text(encoding="utf-8")


def passwords(root: Path) -> dict:
    """The table in README_AFTER_APPROVAL.md: | `file` | `password` |"""
    return dict(re.findall(r"\|\s*`([^`]+\.zip)`\s*\|\s*`([^`]+)`\s*\|", readme(root)))


def checksums(root: Path) -> dict:
    return {f: h for h, f in re.findall(r"^([0-9a-f]{64})\s+(\S.*?)\s*$", readme(root), flags=re.M)}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sizes():
    import requests
    from huggingface_hub import get_token
    r = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main",
                     headers={"Authorization": f"Bearer {get_token()}"}, timeout=60)
    r.raise_for_status()
    return {f["path"]: f.get("size") for f in r.json()}


def curl(f, dst: Path):
    """One file over plain HTTPS, resumed from whatever is on disk; a transfer slower than 1 MB/s for two
    minutes is abandoned and resumed (hf_hub_download stalled for an hour on a finished 20 GiB part)."""
    from urllib.parse import quote
    from huggingface_hub import get_token
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{quote(f)}"
    cfg = f'header = "Authorization: Bearer {get_token()}"\n'           # on stdin: never in the process list
    r = subprocess.run(["curl", "-sS", "-L", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "10",
                        "--speed-limit", "1000000", "--speed-time", "120", "-K", "-", "-o", str(dst), url],
                       input=cfg, text=True, capture_output=True)
    return r.returncode, r.stderr.strip()[-200:]


def download(root: Path, files):
    arc = root / "_archives"
    arc.mkdir(parents=True, exist_ok=True)
    want, size = checksums(root), sizes()

    def one(f):
        p = arc / f
        ok = p.with_name(p.name + ".sha256ok")
        if ok.exists():
            return f, "ok (verified before)"
        for attempt in range(30):
            have = p.stat().st_size if p.exists() else 0
            if size.get(f) and have >= size[f]:
                break
            rc, err = curl(f, p)
            if rc:
                print(f"[download] {f}: curl exit {rc} at {have / 1e9:.1f} GB, retrying ({err})", flush=True)
                time.sleep(15)
        got = sha256(p)
        if got != want.get(f):
            return f, f"CHECKSUM MISMATCH {got[:12]} != {str(want.get(f))[:12]}"
        ok.write_text(got)
        return f, "ok"
    bad = 0
    with ThreadPoolExecutor(3) as ex:
        for f, status in ex.map(one, files):
            print(f"[download] {f}: {status}", flush=True)
            bad += not status.startswith("ok")
    return bad


def seven(args, **kw):
    return subprocess.run([SEVEN, *args], check=True, text=True, capture_output=True, **kw)


def listing(arc: Path, password: str):
    """Paths inside an archive (a split one is opened through its .zip)."""
    out = seven(["l", "-slt", f"-p{password}", str(arc)]).stdout
    return [m for m in re.findall(r"^Path = (.+)$", out, flags=re.M)][1:]      # the first Path is the archive itself


def label_rows(csv_path: Path):
    """MER label files: a name column and a discrete-emotion column, whatever the header calls them."""
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    keys = {k.lower().strip(): k for k in rows[0]}
    name = next(keys[k] for k in keys if k in ("name", "filename", "file", "video", "id"))
    emo = next((keys[k] for k in keys if k in ("discrete", "emotion", "label", "emo", "category")), None)
    val = next((keys[k] for k in keys if k in ("valence", "val")), None)
    return [{"name": r[name].strip(), "label": (r[emo] or "").strip().lower() if emo else "",
             "valence": float(r[val]) if val and r[val] not in ("", None) else None} for r in rows]


def to_wav(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1",
                        "-ar", "16000", "-sample_fmt", "s16", str(tmp)], capture_output=True, text=True)
    if r.returncode or not tmp.exists() or tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        return r.stderr.strip()[-200:] or "empty output"
    tmp.rename(dst)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("step", choices=["download", "download-small", "list", "extract"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None, help="download just these files")
    args = ap.parse_args()
    root, arc = args.root, args.root / "_archives"
    root.mkdir(parents=True, exist_ok=True)
    pw = passwords(root)
    if args.step in ("download", "download-small"):
        files = SMALL + ([] if args.step == "download-small" else TRAIN_PARTS)
        files = [f for f in files if not args.only or f in args.only]
        sys.exit(1 if download(root, files) else 0)
    if args.step == "list":
        for a in ("test-labels.zip", "mer2023test1&2.zip", "mer2023train.zip"):
            if (arc / a).exists():
                paths = listing(arc / a, pw[a])
                vids = [p for p in paths if p.lower().endswith(VIDEO)]
                other = [p for p in paths if not p.lower().endswith(VIDEO)]
                dirs = sorted({str(Path(p).parent) for p in vids})
                print(f"== {a}: {len(paths)} entries, {len(vids)} videos in {len(dirs)} folders "
                      f"({', '.join(dirs[:8])}{' ...' if len(dirs) > 8 else ''})")
                for p in other[:40]:
                    print("   ", p)
        return
    # extract: label files first, then only the labelled videos, then audio
    lab_dir, vid_dir, wav_dir = root / "labels", root / "_videos", root / "audio"
    seven(["x", "-y", f"-p{pw['test-labels.zip']}", f"-o{lab_dir}", str(arc / "test-labels.zip")])
    train_paths = listing(arc / "mer2023train.zip", pw["mer2023train.zip"])
    test_paths = listing(arc / "mer2023test1&2.zip", pw["mer2023test1&2.zip"])
    csvs = [p for p in train_paths + test_paths if p.lower().endswith(".csv")]
    for p in csvs:                                  # label files shipped inside the data archives
        src = "mer2023train.zip" if p in train_paths else "mer2023test1&2.zip"
        seven(["x", "-y", f"-p{pw[src]}", f"-o{lab_dir}", str(arc / src), p])
    labels = {}
    for c in sorted(lab_dir.rglob("*.csv")):
        split = c.stem.lower().replace("-label", "").replace("_label", "")
        for r in label_rows(c):
            if r["label"]:
                labels[r["name"]] = dict(r, split=split, label_file=str(c.relative_to(lab_dir)))
    print(f"[labels] {len(labels)} labelled names from {len(list(lab_dir.rglob('*.csv')))} csv files: "
          f"{json.dumps({s: sum(v['split'] == s for v in labels.values()) for s in sorted({v['split'] for v in labels.values()})})}",
          flush=True)
    by_stem = {}
    for src, paths in (("mer2023train.zip", train_paths), ("mer2023test1&2.zip", test_paths)):
        for p in paths:
            if p.lower().endswith(VIDEO):
                by_stem.setdefault(Path(p).stem, (src, p))
    missing = [n for n in labels if n not in by_stem]
    print(f"[labels] {len(labels) - len(missing)} have a video, {len(missing)} do not "
          f"(e.g. {missing[:3]})", flush=True)
    for src in ("mer2023test1&2.zip", "mer2023train.zip"):
        todo = [by_stem[n][1] for n in labels if n in by_stem and by_stem[n][0] == src
                and not (wav_dir / f"{n}.wav").exists()]
        if not todo:
            continue
        lst = root / f"_extract_{src.split('.')[0].replace('&', '_')}.txt"
        lst.write_text("\n".join(todo) + "\n", encoding="utf-8")
        print(f"[extract] {len(todo)} videos from {src}", flush=True)
        seven(["x", "-y", f"-p{pw[src]}", f"-o{vid_dir}", str(arc / src), f"@{lst}"])
    jobs = [(vid_dir / by_stem[n][1], wav_dir / f"{n}.wav") for n in labels if n in by_stem]
    jobs = [(s, d) for s, d in jobs if not d.exists()]
    failed = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for (s, d), err in zip(jobs, ex.map(lambda j: to_wav(*j), jobs)):
            if err:
                failed[d.stem] = err
    rows = [dict(v, audio=str(wav_dir / f"{n}.wav")) for n, v in sorted(labels.items())
            if (wav_dir / f"{n}.wav").exists()]
    with (root / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (root / "extract_failed.json").write_text(json.dumps(failed, indent=1))
    counts = {}
    for r in rows:
        counts.setdefault(r["split"], {}).setdefault(r["label"], 0)
        counts[r["split"]][r["label"]] += 1
    print(json.dumps({"clips": len(rows), "failed": len(failed), "missing_video": len(missing), "by_split": counts}))
    (root / ".fetched").write_text(json.dumps({"files": len(rows)}))       # emotion/build_manifest.py reads it
    print("MER2023_EXTRACTED", flush=True)


if __name__ == "__main__":
    main()
