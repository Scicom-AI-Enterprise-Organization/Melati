"""Whether the box's Hugging Face token can read a (possibly gated) repo: lists a few files and fetches the smallest
one's metadata. RUNS ON THE BOX.

    python scripts/hf_can_read.py --repo confit/audioset-16khz-wds --repo-type dataset --prefix 2m/test/
"""
import argparse

from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--repo-type", default="dataset")
ap.add_argument("--prefix", default="")
a = ap.parse_args()
api = HfApi()
print("as", api.whoami()["name"])
files = [f for f in api.list_repo_files(a.repo, repo_type=a.repo_type) if f.startswith(a.prefix)]
print(f"{len(files)} files under {a.prefix!r}: {files[:4]}")
try:
    m = get_hf_file_metadata(hf_hub_url(a.repo, files[0], repo_type=a.repo_type))
    print(f"readable: {files[0]} {m.size / 2 ** 20:.0f} MiB")
except Exception as e:                                        # a gated repo whose terms were not accepted
    print(f"NOT readable: {type(e).__name__}: {str(e)[:200]}")
