#!/usr/bin/env python3
"""What does a Hugging Face dataset hold? Configs, their columns and sizes, from the card's
`dataset_info` (no data is downloaded), plus one row of each requested config.

    python asr/scripts/inspect_hf_dataset.py Scicom-intl/TTS-Clean44k [--rows kannada gujarati]
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--rows", nargs="*", default=[], help="configs to read one row from")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    import yaml
    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    info = api.dataset_info(args.repo)
    print(f"{args.repo}: private={info.private} license={(info.card_data or {}).get('license')}")
    card = open(hf_hub_download(args.repo, "README.md", repo_type="dataset", token=tok), encoding="utf-8").read()
    meta = yaml.safe_load(card.split("---")[1]) if card.startswith("---") else {}
    infos = meta.get("dataset_info") or []
    infos = infos if isinstance(infos, list) else [infos]
    for d in infos:
        feats = [f["name"] for f in d.get("features") or []]
        rows = sum(s.get("num_examples", 0) for s in d.get("splits") or [])
        gb = (d.get("dataset_size") or 0) / 1e9
        print(f"  {d.get('config_name', 'default'):28s} rows {rows:>9,}  {gb:7.1f} GB  cols {feats}")
    if not infos:
        files = api.list_repo_files(args.repo, repo_type="dataset")
        tops = sorted({f.split("/")[0] for f in files})
        print(f"  no dataset_info; top-level entries: {tops[:60]}")
    for cfg in args.rows:
        from datasets import Audio, load_dataset
        d = load_dataset(args.repo, cfg, split="train", streaming=True, token=tok)
        try:
            d = d.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
        r = next(iter(d))
        print(f"  [{cfg}] " + json.dumps({k: (v if not isinstance(v, (dict, bytes)) else "<audio>")
                                          for k, v in r.items()}, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
