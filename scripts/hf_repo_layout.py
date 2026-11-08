"""A Hugging Face repo's files grouped by folder: count and size of each, to see what a dataset mirror holds before
downloading it. RUNS ON THE BOX.

    python scripts/hf_repo_layout.py --repo agkphysics/AudioSet
"""
import argparse
from collections import defaultdict

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--repo-type", default="dataset")
ap.add_argument("--depth", type=int, default=2)
a = ap.parse_args()
info = HfApi().dataset_info(a.repo, files_metadata=True) if a.repo_type == "dataset" else HfApi().model_info(
    a.repo, files_metadata=True)
groups = defaultdict(lambda: [0, 0])
for s in info.siblings:
    g = "/".join(s.rfilename.split("/")[:a.depth]) if "/" in s.rfilename else "(top)"
    groups[g][0] += 1
    groups[g][1] += s.size or 0
for g, (n, b) in sorted(groups.items()):
    print(f"{g:40s} {n:5d} files {b / 2 ** 30:8.1f} GiB")
print(f"total {sum(b for _, b in groups.values()) / 2 ** 30:.1f} GiB; gated={info.gated}; license={(info.card_data or {}).get('license') if info.card_data else None}")
