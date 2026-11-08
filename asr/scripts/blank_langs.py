#!/usr/bin/env python3
"""The language the BASE model detects on every blank (voice-free) clip of the training mixes. RUNS ON THE BOX.

A blank clip's target is "no words". At inference that target follows whatever language token the model
detects on the clip, so it is learned where it will be used only if training puts the same token there.
Drawn at random from the mix's 89 languages (finetune_whisper.py before this), "silence after <|en|> ->
stop" was seen about one time in 89, and with the task token stated a fixed fine-tune wrote "you" over 7 of
42 silent clips. This writes {audio path: detected language} for finetune_whisper.py --blank-lang-map,
and prints what base detects on voice-free audio.

    python asr/scripts/blank_langs.py --mixes $MELATI_DATA/asr/mixes/all.jsonl \\
        --out $MELATI_DATA/asr/mixes/blank_langs.json
"""
import argparse, json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixes", nargs="+", type=Path, required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    paths = sorted({r["audio_filepath"] for m in args.mixes for r in map(json.loads, m.open(encoding="utf-8"))
                    if not (r.get("text") or "").strip()})
    print(f"{len(paths)} blank clips", flush=True)
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    torch.set_grad_enabled(False)
    out = {}
    for i in range(0, len(paths), args.batch_size):
        batch, keep = [], []
        for p in paths[i:i + args.batch_size]:
            try:
                x, sr = sf.read(p, dtype="float32")
            except Exception:
                continue
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            batch.append(x); keep.append(p)
        if not batch:
            continue
        feats = proc(batch, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        ids = model.detect_language(input_features=feats)
        for p, t in zip(keep, proc.tokenizer.convert_ids_to_tokens(ids.tolist())):
            out[p] = t.strip("<|>")
        if (i // args.batch_size) % 100 == 0:
            print(f"  {len(out)}/{len(paths)}", flush=True)
    args.out.write_text(json.dumps(out))
    c = Counter(out.values())
    print(f"{len(out)} clips -> {args.out}; base detects: {dict(c.most_common(12))}", flush=True)


if __name__ == "__main__":
    main()
