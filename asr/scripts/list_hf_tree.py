#!/usr/bin/env python3
"""List one folder of a Hugging Face dataset repo, grouped by name prefix, with file counts and sizes.

    python asr/scripts/list_hf_tree.py Scicom-intl/TTS-Clean44k labeled44k
"""
import argparse, os, re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("path")
    ap.add_argument("--depth", type=int, default=2, help="path components after `path` to group by")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    groups = defaultdict(lambda: [0, 0])
    n = 0
    for e in api.list_repo_tree(args.repo, path_in_repo=args.path, repo_type="dataset", recursive=True):
        size = getattr(e, "size", None)
        if size is None:
            continue
        n += 1
        rel = e.path[len(args.path) + 1:]
        parts = rel.split("/")
        key = "/".join(parts[: args.depth]) if len(parts) > args.depth else re.sub(r"[-_]?\d+(-of-\d+)?\.\w+$", "*", rel)
        groups[key][0] += 1
        groups[key][1] += size
    print(f"{n} files under {args.path}")
    for k, (c, b) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:80]:
        print(f"  {k:60s} {c:6d} files {b / 1e9:8.2f} GB")


if __name__ == "__main__":
    main()
