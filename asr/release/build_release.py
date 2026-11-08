#!/usr/bin/env python3
"""Assemble the released checkpoint from a run's merged model, check it, and optionally upload it. RUNS ON THE BOX.

The weights and tokenizer are the run's `merged/` as saved. The generation config is base whisper-large-v3's
with three changes, so that `transformers` decodes the way the benchmark measured and vLLM serves:

  task "transcribe"          stated, never predicted: a fine-tune left to predict the task token can translate
  begin_suppress_tokens None the default forbids <|endoftext|> as the first token, i.e. forbids an empty
                             transcript, which is the one answer this model was trained to give on silence
  suppress_tokens None       vLLM suppresses nothing either

Before anything is uploaded the folder is loaded back and run on voice-free clips and speech clips from
dump_arms.py's arrays: it must write nothing on silence and something on speech.

    python asr/release/build_release.py --run $MELATI_DATA/asr/runs2/<run> --out $MELATI_DATA/release/<name> \\
        --arms-dir $MELATI_DATA/asr/arms_npz [--repo Scicom-intl/<name> --push]
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from metrics import normalise  # noqa: E402

BASE = "openai/whisper-large-v3"


def generation_config(base_dir: Path) -> dict:
    g = json.loads((base_dir / "generation_config.json").read_text())
    g.update(task="transcribe", begin_suppress_tokens=None, suppress_tokens=None, forced_decoder_ids=None)
    return g


def smoke(folder: Path, arms_dir: Path, n: int = 24) -> dict:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(folder)
    model = WhisperForConditionalGeneration.from_pretrained(folder, dtype=torch.float16).to("cuda").eval()
    out = {}
    for arm in ("silence", "music", "fleurs"):
        z = np.load(arms_dir / f"{arm}.npz")
        a, off = z["audio"], z["offsets"]
        idx = np.linspace(0, len(off) - 2, num=min(n, len(off) - 1)).astype(int)
        waves = [a[off[i]:off[i + 1]] for i in idx]
        feats = proc(waves, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=200)           # the shipped config, nothing passed
        hyps = proc.batch_decode(ids, skip_special_tokens=True)
        out[arm] = {"n": len(hyps), "with_words": sum(bool(normalise(h)) for h in hyps), "examples": hyps[:3]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms-dir", type=Path, required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    base_dir = Path(snapshot_download(BASE, allow_patterns=["generation_config.json"]))
    src = args.run / "merged"
    args.out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "generation_config.json":
            shutil.copy2(f, args.out / f.name)
    (args.out / "generation_config.json").write_text(json.dumps(generation_config(base_dir), indent=2) + "\n")
    s = smoke(args.out, args.arms_dir)
    print(json.dumps(s, ensure_ascii=False, indent=1), flush=True)
    assert s["silence"]["with_words"] == 0, "writes words on silence with the shipped config"
    assert s["fleurs"]["with_words"] >= s["fleurs"]["n"] - 1, "writes nothing on speech"
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=str(args.out), commit_message="Weights and generation config",
                          allow_patterns=["*.json", "*.txt", "*.safetensors"])
        print(f"pushed {args.out} -> https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
