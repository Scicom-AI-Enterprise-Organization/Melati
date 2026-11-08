"""Fetch a Hugging Face repo's files matching a pattern, many at a time, each retried on failure: huggingface-cli
downloads one file after another, and one stalled transfer then holds up the rest. RUNS ON THE BOX.

    python scripts/hf_fetch.py --repo agkphysics/AudioSet --include "data/eval/*" --out $MELATI_DATA/audioset/agk \
        [--verify]
"""
import argparse
import fnmatch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--repo-type", default="dataset")
ap.add_argument("--include", nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--tries", type=int, default=5)
ap.add_argument("--verify", action="store_true", help="check every local file's sha256, refetching any that differs")
a = ap.parse_args()
info = HfApi().dataset_info(a.repo, files_metadata=True)
files = [s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in a.include)]
def intact(s):
    """The local copy is whole: its size, and with --verify its sha256 against the repo's LFS record."""
    f = a.out / s.rfilename
    if not f.exists() or f.stat().st_size != s.size:
        return False
    if a.verify and s.lfs:
        import hashlib
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != s.lfs.sha256:
            print(f"corrupt, fetching again: {s.rfilename}", flush=True)
            f.unlink()
            return False
    return True


todo = [s for s in files if not intact(s)]
print(f"{len(files)} files ({sum(s.size or 0 for s in files) / 2 ** 30:.1f} GiB), {len(todo)} to fetch", flush=True)


def one(s):
    for k in range(a.tries):
        try:
            hf_hub_download(a.repo, s.rfilename, repo_type=a.repo_type, local_dir=a.out, etag_timeout=60)
            return s.rfilename, s.size
        except Exception as e:                                     # a dropped connection: wait and try again
            if k == a.tries - 1:
                raise
            print(f"retry {s.rfilename}: {type(e).__name__}", flush=True)
            time.sleep(10 * (k + 1))


t0, got = time.time(), 0
with ThreadPoolExecutor(a.workers) as ex:
    for k, f in enumerate(as_completed([ex.submit(one, s) for s in todo]), 1):
        got += f.result()[1] or 0
        if k % 10 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} files, {got / 2 ** 30:.1f} GiB, {got / (time.time() - t0) / 2 ** 20:.0f} MiB/s",
                  flush=True)
print("HF_FETCH_DONE", flush=True)
