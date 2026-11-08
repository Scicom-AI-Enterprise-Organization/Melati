#!/usr/bin/env python3
"""What does the decoder SEE in training? One clip through ClipDataset + Collator, then the shift
Hugging Face applies when `labels` is given without `decoder_input_ids`
(WhisperForConditionalGeneration.forward -> shift_tokens_right: labels moved right by one, the
decoder start token in front, every -100 replaced by pad). Prints both rows token by token, so a
prefix masked out of the LOSS can be checked against what is fed as INPUT.

    python asr/train/check_labels.py --mix $MELATI_DATA/asr/mixes/all_cv22_r50.jsonl
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", type=Path, required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    from transformers import WhisperConfig, WhisperProcessor
    from transformers.models.whisper.modeling_whisper import shift_tokens_right
    from finetune_whisper import ClipDataset, Collator
    proc = WhisperProcessor.from_pretrained(args.model)
    cfg = WhisperConfig.from_pretrained(args.model)
    all_rows = [json.loads(line) for line in args.mix.open()]
    speech = [r for r in all_rows if r.get("text") and r.get("lang") not in ("", "en", None)][: args.n]
    blank = [r for r in all_rows if not (r.get("text") or "").strip()][:1]
    rows = speech + blank
    ds = ClipDataset(all_rows, proc)            # blank languages come from the whole mix
    ds.rows = rows
    batch = Collator(proc)([ds[i] for i in range(len(rows))])
    if "decoder_input_ids" in batch:            # the collator feeds the prefix itself
        dec_in = batch["decoder_input_ids"]
        print(f"decoder_input_ids from the collator; blank clips draw from {len(ds.blank_langs)} languages")
    else:                                       # Hugging Face builds them by shifting the labels
        dec_in = shift_tokens_right(batch["labels"], cfg.pad_token_id, cfg.decoder_start_token_id)
        print("decoder_input_ids = shift_tokens_right(labels)")
    tok = proc.tokenizer
    print(f"pad_token_id {cfg.pad_token_id} ({tok.convert_ids_to_tokens(cfg.pad_token_id)}), "
          f"decoder_start {cfg.decoder_start_token_id} ({tok.convert_ids_to_tokens(cfg.decoder_start_token_id)})")
    for b in range(len(rows)):
        print(f"\nclip lang={rows[b].get('lang')!r} text={(rows[b].get('text') or '')[:60]!r}")
        print(f"{'pos':>3}  {'decoder input':<24} {'label (loss target)':<24}")
        for p in range(8):
            lab = int(batch["labels"][b, p])
            print(f"{p:3d}  {tok.convert_ids_to_tokens(int(dec_in[b, p])):<24} "
                  f"{'(ignored)' if lab == -100 else tok.convert_ids_to_tokens(lab):<24}")


if __name__ == "__main__":
    main()
