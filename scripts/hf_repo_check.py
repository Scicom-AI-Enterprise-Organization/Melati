#!/usr/bin/env python3
"""What a Hugging Face model repo holds, as the public sees it, and an optional cleanup. RUNS ON THE BOX.
--squash rewrites the repo's history into its current commit, so weights an earlier revision held can no longer be
downloaded by revision (the first non-verbal release trained on data whose licence did not allow it).

    python scripts/hf_repo_check.py Scicom-intl/Melati-NonVerbal-Whisper-Large-V3 [--delete-folder __pycache__] [--squash]
"""
import argparse
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--delete-folder", default=None)
ap.add_argument("--delete-files", nargs="*", default=None, help="files an earlier release left that this one does not use")
ap.add_argument("--squash", default=None, metavar="MESSAGE", help="squash the history into one commit with this message")
args = ap.parse_args()
api = HfApi()
if args.delete_folder:
    api.delete_folder(args.delete_folder, repo_id=args.repo, repo_type="model",
                      commit_message=f"remove {args.delete_folder}")
if args.delete_files:
    api.delete_files(repo_id=args.repo, delete_patterns=args.delete_files, repo_type="model",
                     commit_message=f"remove {', '.join(args.delete_files)}: the code is in the *_melati_head.py files")
if args.squash:
    before = len(api.list_repo_commits(args.repo, repo_type="model"))
    api.super_squash_history(args.repo, repo_type="model", commit_message=args.squash)
    print(f"squashed {before} commits into {len(api.list_repo_commits(args.repo, repo_type='model'))}")
info = api.model_info(args.repo, files_metadata=True)
print("private:", info.private, "| gated:", info.gated, "| license:", (info.card_data or {}).get("license"),
      (info.card_data or {}).get("license_name"))
for f in info.siblings:
    print(f"  {f.rfilename:28s} {f.size or 0:>10,} bytes")
