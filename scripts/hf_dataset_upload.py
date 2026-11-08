#!/usr/bin/env python3
"""Upload Melati's dataset folder to Hugging Face, resumably, and publish it once it reads back. RUNS ON THE BOX.

    python scripts/hf_dataset_upload.py --folder $MELATI_DATA/podcast/hf --repo Scicom-intl/Melati
    python scripts/hf_dataset_upload.py --repo Scicom-intl/Melati --public      # after check_hf_dataset.py --repo
    python scripts/hf_dataset_upload.py --folder ... --repo Scicom-intl/Melati --one-commit --message "..."

The repo is created private and stays private until --public: a half-uploaded dataset is never visible.
upload_large_folder keeps its progress in <folder>/.cache/huggingface, so a re-run after a failure sends only
what is missing. --one-commit replaces every file at once instead (upload_folder: the blobs go up first and one
commit swaps them in), for a re-pack whose shards change their columns: upload_large_folder commits in batches, and
a reader in between would meet old and new shards with two schemas. A failed --one-commit re-run sends only the blobs
the repo does not hold yet.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--folder", type=Path, default=None)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--public", action="store_true", help="make the repo public (nothing is uploaded)")
ap.add_argument("--one-commit", action="store_true", help="every file in a single commit (a re-pack)")
ap.add_argument("--message", default=None, help="the commit message, with --one-commit")
args = ap.parse_args()
api = HfApi()
if args.public:
    api.update_repo_settings(args.repo, repo_type="dataset", private=False)
else:
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    patterns = ["README.md", "export_config.json", "stats.json", "figures/*.png", "data/*.parquet"]
    if args.one_commit:
        api.upload_folder(repo_id=args.repo, folder_path=str(args.folder), repo_type="dataset",
                          allow_patterns=patterns, commit_message=args.message or "Re-pack every shard")
    else:
        api.upload_large_folder(repo_id=args.repo, folder_path=str(args.folder), repo_type="dataset",
                                allow_patterns=patterns, num_workers=args.workers, print_report_every=300)
info = api.dataset_info(args.repo, files_metadata=True)
files = [f for f in info.siblings if f.rfilename.startswith("data/")]
print(f"{args.repo}: private={info.private}, {len(files)} data files, "
      f"{sum(f.size or 0 for f in files) / 2 ** 30:.1f} GiB, {len(info.siblings)} files in all")
print("HF_UPLOAD_DONE")
