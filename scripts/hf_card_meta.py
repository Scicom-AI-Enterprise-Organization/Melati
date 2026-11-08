#!/usr/bin/env python3
"""A Hugging Face repo's card metadata (licence, languages, tags) and the start of its card. RUNS ON THE BOX.

    python scripts/hf_card_meta.py malaysia-ai/malaysian-podcast-youtube Scicom-intl/Malaysian-Emilia --type dataset
"""
import argparse
from huggingface_hub import HfApi, hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("repos", nargs="+")
ap.add_argument("--type", default="dataset")
ap.add_argument("--chars", type=int, default=1500)
args = ap.parse_args()
api = HfApi()
for r in args.repos:
    info = api.dataset_info(r) if args.type == "dataset" else api.model_info(r)
    print(f"== {r}: private={info.private} gated={info.gated} license={(info.card_data or {}).get('license')}")
    try:
        text = open(hf_hub_download(r, "README.md", repo_type=args.type)).read()
        print(text[: args.chars])
    except Exception as e:  # noqa: BLE001
        print(f"  no README: {type(e).__name__}")
