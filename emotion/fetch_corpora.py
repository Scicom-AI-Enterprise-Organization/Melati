#!/usr/bin/env python3
"""Download the openly available emotional-speech corpora for Melati's own emotion classifier. RUNS ON THE BOX.

The list is EmoBox's 32 corpora (github.com/emo-box/EmoBox, which ships folds and label maps but no audio) plus
NonverbalTTS and Expresso, minus what needs a signed agreement or an account we do not hold:

  needs Scicom to sign   MSP-Podcast (UT Dallas), IEMOCAP (USC), ESD (NUS licence form)
  gated, accepted        MER2023 (Hugging Face, non-commercial academic): emotion/fetch_mer2023.py
  needs an account       CASIA (Baidu AI Studio), M3ED (Baidu Netdisk), SAVEE, ASVP-ESD and JL-Corpus (Kaggle)
  in CAMEO already       CREMA-D, CaFE, EMNS, Emozionalmente, eNTERFACE, JL-Corpus, MESD, nEMO, Oreau, PAVOQUE,
                         RAVDESS, RESD, SUBESCO (amu-cai/CAMEO, one download)

Each corpus lands in <root>/<name>/ and is skipped when its marker file exists, so a rerun fetches only what
failed. A page whose archive links are not fixed is read for them. Licences are the sources'; nearly all are
research or non-commercial, which Melati's CC BY-NC release respects. --sizes says what each would download and
fetches nothing (a git repository's size as GitHub reports it, which leaves out LFS objects; 0 where a host gives
no length).

    python emotion/fetch_corpora.py --root $MELATI_DATA/emo/raw [--only cameo meld] [--sizes]
"""
import argparse, json, re, shutil, subprocess, sys, tarfile, zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests

UA = {"User-Agent": "Mozilla/5.0 (melati corpus fetch)"}


def get(url, dst: Path, timeout=600):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, headers=UA, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    tmp.rename(dst)
    return dst


def unpack(arc: Path, into: Path):
    into.mkdir(parents=True, exist_ok=True)
    name = arc.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            z.extractall(into)
    elif name.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2")):
        with tarfile.open(arc) as t:
            t.extractall(into)
    elif name.endswith((".rar", ".7z")):
        subprocess.run([shutil.which("7z") or "7zz", "x", "-y", f"-o{into}", str(arc)], check=True,
                       stdout=subprocess.DEVNULL)
    else:
        raise ValueError(f"cannot unpack {arc}")


def length(url):
    """A download's size from its headers, without fetching it (0 when the host does not say)."""
    with requests.get(url, headers=UA, stream=True, timeout=60, allow_redirects=True) as r:
        return int(r.headers.get("content-length") or 0)


def hf_sizes(repo):
    from huggingface_hub import HfApi
    return {f.rfilename: f.size or 0 for f in HfApi().dataset_info(repo, files_metadata=True).siblings}


def links(page_url, pattern):
    html = requests.get(page_url, headers=UA, timeout=60).text
    return sorted({urljoin(page_url, h) for h in re.findall(r'href=["\']([^"\']+)["\']', html) if re.search(pattern, h)})


def hf(repo):
    def run(dst: Path):
        from huggingface_hub import snapshot_download
        snapshot_download(repo, repo_type="dataset", local_dir=str(dst), max_workers=8)
    run.size = lambda: sum(hf_sizes(repo).values())
    return run


def git(url):
    def run(dst: Path):
        if not (dst / ".git").exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dst)], check=True)
        subprocess.run(["git", "-C", str(dst), "lfs", "pull"], check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        for z in list(dst.rglob("*.zip")) + list(dst.rglob("*.rar")):
            unpack(z, z.parent / z.stem)
    repo = url.split("github.com/", 1)[1]
    run.size = lambda: 1024 * int(requests.get(f"https://api.github.com/repos/{repo}", headers=UA, timeout=60)
                                  .json().get("size") or 0)
    return run


def archives(page, pattern, per_archive=False):
    """Every archive a page links to. per_archive unpacks each into its own folder (named after the archive):
    EmoV-DB's speakers share file names, and one folder for all of them kept only the last speaker's takes."""
    def run(dst: Path):
        found = links(page, pattern)
        if not found:
            raise RuntimeError(f"no archive links on {page}")
        for u in found:
            arc = get(u, dst / "_archives" / Path(u.split("?")[0]).name)
            unpack(arc, dst / arc.name.split(".")[0] if per_archive else dst)
    run.size = lambda: sum(length(u) for u in links(page, pattern))
    return run


def direct(*urls):
    def run(dst: Path):
        for u in urls:
            arc = get(u, dst / "_archives" / Path(u.split("?")[0]).name)
            unpack(arc, dst)
    run.size = lambda: sum(length(u) for u in urls)
    return run


def hf_file(repo, filename):
    """One archive from a Hugging Face dataset repo, unpacked, and any archives inside it unpacked too."""
    def run(dst: Path):
        from huggingface_hub import hf_hub_download
        arc = Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(dst / "_archives")))
        unpack(arc, dst)
        for inner in sorted(p for p in dst.rglob("*") if p.name.lower().endswith((".tar.gz", ".tgz"))
                            and "_archives" not in p.parts):
            unpack(inner, inner.parent)
    run.size = lambda: hf_sizes(repo).get(filename, 0)
    return run


def zenodo(record):
    def run(dst: Path):
        meta = requests.get(f"https://zenodo.org/api/records/{record}", headers=UA, timeout=60).json()
        for f in meta.get("files", []):
            u = f["links"]["self"]
            arc = get(u, dst / "_archives" / f["key"])
            if f["key"].lower().endswith((".zip", ".tar.gz", ".tgz", ".rar", ".7z")):
                unpack(arc, dst)
    run.size = lambda: sum(f.get("size") or 0 for f in requests.get(f"https://zenodo.org/api/records/{record}",
                                                                     headers=UA, timeout=60).json().get("files", []))
    return run


CORPORA = {
    "cameo": hf("amu-cai/CAMEO"),
    "nonverbaltts": hf("deepvk/NonverbalTTS"),
    "expresso": hf("ylacombe/expresso"),
    "meld": hf_file("declare-lab/MELD", "MELD.Raw.tar.gz"),   # the umich host's certificate fails
    "emovdb": archives("https://www.openslr.org/115/", r"resources/115/.*\.tar\.gz$", per_archive=True),
    "emodb": direct("http://emodb.bilderbar.info/download/download.zip"),
    "tess": archives("https://tspace.library.utoronto.ca/handle/1807/24487", r"bitstream/.*\.(zip|wav)"),
    "shemo": git("https://github.com/mansourehk/ShEMO"),
    "ased": git("https://github.com/wgwangang/ASED_V1"),
    "turevdb": git("https://github.com/Xeonen/TurEV-DB"),
    "urdu": git("https://github.com/siddiquelatif/urdu-dataset"),
    "aesdd": archives("http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/", r"\.(zip|rar)$"),
    "emovo": archives("http://voice.fub.it/activities/corpora/emovo/index.html", r"\.(zip|rar)$"),
    "polish": archives("https://mostwiedzy.pl/en/open-research-data/emotions-in-polish-speech-recordings,11190523461146169-0",
                       r"(download|\.zip)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--sizes", action="store_true", help="say what each corpus would download, fetch nothing")
    args = ap.parse_args()
    if args.sizes:
        total = 0
        for name, fetch in CORPORA.items():
            if args.only and name not in args.only:
                continue
            try:
                n = fetch.size()
                total += n
                print(f"[size] {name:14s} {n / 1e9:8.2f} GB", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[size] {name:14s}  unknown ({type(e).__name__}: {str(e)[:120]})", flush=True)
        print(f"[size] total          {total / 1e9:8.2f} GB")
        return
    status = {}
    for name, fetch in CORPORA.items():
        if args.only and name not in args.only:
            continue
        dst = args.root / name
        marker = dst / ".fetched"
        if marker.exists():
            status[name] = "ok (cached)"
            continue
        try:
            fetch(dst)
            n = sum(1 for p in dst.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".mp4", ".parquet"))
            marker.write_text(json.dumps({"files": n}))
            status[name] = f"ok ({n} audio/parquet files)"
        except Exception as e:  # noqa: BLE001
            status[name] = f"FAILED {type(e).__name__}: {str(e)[:160]}"
        print(f"[fetch] {name}: {status[name]}", flush=True)
    (args.root / "fetch_status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    sys.exit(main())
