#!/usr/bin/env python3
"""Language identification on FLEURS, per checkpoint: does a fine-tune still name the language it hears?
RUNS ON THE BOX (a GPU).

Whisper names the language from the start token alone (`detect_language`, what generate() calls when no
language is given), and a fine-tune that masks that token out of its loss lets the guess drift (asr/train/
finetune_whisper.py, top). This asks only that question: one encoder pass and one decoder step per clip,
no transcription. Per model: accuracy, per-language accuracy, and what the misses became.

    python asr/lid_accuracy.py --models base=openai/whisper-large-v3 fix=$MELATI_DATA/asr/runs2/<run>/merged \\
        --fleurs-cache $MELATI_DATA/asr/bench/fleurs_sample.parquet --out $MELATI_DATA/results/asr/lid_accuracy.json
"""
import argparse, io, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=path-or-hub-id")
    ap.add_argument("--fleurs-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from run_benchmark import load_fleurs
    ds = load_fleurs(20, 0, args.fleurs_cache)
    rows = [ds[i] for i in range(len(ds))]
    waves = [sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0] for r in rows]
    langs = [r.get("lang") or (r.get("meta") or {}).get("lang") for r in rows]
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    torch.set_grad_enabled(False)
    for spec in args.models:
        name, path = spec.split("=", 1)
        proc = WhisperProcessor.from_pretrained(path)
        model = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float16).to("cuda").eval()
        got = []
        for i in range(0, len(waves), args.batch_size):
            feats = proc(waves[i:i + args.batch_size], sampling_rate=16000, return_tensors="pt").input_features
            ids = model.detect_language(input_features=feats.to("cuda", torch.float16))
            got += [t.strip("<|>") for t in proc.tokenizer.convert_ids_to_tokens(ids.tolist())]
        per, wrong = defaultdict(lambda: [0, 0]), Counter()
        for l, g in zip(langs, got):
            per[l][0] += g == l
            per[l][1] += 1
            if g != l:
                wrong[g] += 1
        acc = sum(c for c, _ in per.values()) / len(got)
        res[name] = {"model": path, "accuracy": round(acc, 4), "n": len(got),
                     "per_lang": {k: round(c / n, 3) for k, (c, n) in sorted(per.items())},
                     "wrong_as": dict(wrong.most_common(10))}
        print(f"{name:28s} LID accuracy {acc:.3f}   misses mostly -> {dict(wrong.most_common(5))}", flush=True)
        del model
        torch.cuda.empty_cache()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
