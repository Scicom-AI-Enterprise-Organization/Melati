#!/usr/bin/env python3
"""Fine-tune whisper-large-v3 to hallucinate less WITHOUT losing accuracy.

Both halves matter and they pull against each other: a model that emits nothing scores
perfectly on the non-speech arms and is useless. So the target for a blank clip is the empty
transcript, the target for a positive clip is its text, and the mix ratio between them is the
sweep's main factor (`train/build_mix.py`).

  --method lora   PEFT adapters on the attention projections AND the MLP (fc1/fc2, where
                  most of a Whisper block's parameters live). Cheap enough to sweep.
  --method full   every weight. One or two runs, for the mix that wins the sweep.

Target format follows the same contract as the rest of the Whisper family:

    <|startoftranscript|><|LANG|><|transcribe|><|notimestamps|> TEXT<|endoftext|>

A blank clip gets that prefix with no TEXT, which is what teaches "this audio has no words"
rather than "emit something short". Prompt tokens are masked to -100 so loss falls only on the
text and the final EOT.

The prefix is fed as DECODER INPUT, explicitly (`decoder_input_ids`). The earlier version masked
it in `labels` and left the inputs to Hugging Face, which builds them by shifting `labels` right
and replacing every -100 with the pad token: the decoder then trained on
`<|startoftranscript|> pad pad pad pad TEXT`, never saw a language token, and met one for the
first time at inference (asr/train/check_labels.py prints both rows). That is the multilingual
regression of the first Melati sweep, measured clip by clip by asr/diagnose_regression.py:
Kannada and Telugu written in Tamil script on every FLEURS clip, and the same code produced the
Whisper-Hallucination paper's FLEURS loss.

Language identification is the token at position 1, predicted from <|startoftranscript|> alone,
and masked out of the loss it is free to drift: LoRA moves the shared weights and the prediction
moves with them. Fed the prefix, the model OBEYS the language token, so a drifted guess turns an
Arabic clip into romanised Arabic and a Yoruba one into English (the first fixed sweep: FLEURS
failures 172 -> 707). `--lid-loss` decides what happens there:
  none     masked, as before
  label    supervised with the clip's own language (speech clips; a blank clip's is random, masked)
  distill  held to the BASE model's distribution over the language tokens (KL), so identification
           stays what it was for every language, including those the mix does not contain; base is
           the same weights with the LoRA adapter off, or a frozen copy for a full fine-tune

Blank clips have no language, and once the prefix is visible the one they are given matters: if
it were always <|en|>, "no words here, stop" would be learned after <|en|> only, while on silence
Whisper detects all sorts of languages. So a blank clip draws its language from the positive
clips' languages, fixed per clip (a hash of its path), so a run is reproducible.

    .venv_train/bin/python train/finetune_whisper.py --mix train/mixes/plus_synth.jsonl \
        --method lora --out runs/lora_plus_synth
"""
import argparse, json, os, random, zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SR = 16000


class ClipDataset(torch.utils.data.Dataset):
    """Reads a mix jsonl. Audio is loaded lazily -- the corpus is 111 h and does not fit in RAM."""

    def __init__(self, rows, processor, max_label_len=200, default_lang="en", lid_label=False, blank_lang_map=None):
        self.rows = rows
        self.proc = processor
        self.tok = processor.tokenizer
        self.feat = processor.feature_extractor
        self.max_label_len = max_label_len
        self.default_lang = default_lang
        self.lid_label = lid_label
        self.blank_lang_map = blank_lang_map or {}       # audio path -> the language base detects on it
        langs = sorted({(r.get("lang") or "").strip() for r in rows if (r.get("text") or "").strip()} - {""})
        self.blank_langs = [l for l in langs if self._known(l)] or [default_lang]

    def _known(self, lang):
        try:
            self.tok.set_prefix_tokens(language=lang, task="transcribe", predict_timestamps=False)
            return True
        except Exception:                      # a language Whisper has no token for
            return False

    def lang_of(self, r):
        lang = (r.get("lang") or "").strip()
        if not (r.get("text") or "").strip():
            # blank: the language base detects on it (what inference will put there, asr/scripts/blank_langs.py),
            # else one from the mix, fixed per clip
            got = self.blank_lang_map.get(r["audio_filepath"])
            if got and self._known(got):
                return got
            return self.blank_langs[zlib.crc32(r["audio_filepath"].encode()) % len(self.blank_langs)]
        return lang if lang and self._known(lang) else self.default_lang

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf
        r = self.rows[i]
        try:
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
        except Exception:
            return None
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        feats = self.feat(x, sampling_rate=SR, return_tensors="np").input_features[0]

        self.tok.set_prefix_tokens(language=self.lang_of(r), task="transcribe", predict_timestamps=False)
        n_prefix = len(self.tok.prefix_tokens)          # <|startoftranscript|><|xx|><|transcribe|><|notimestamps|>
        ids = self.tok(r.get("text") or "", add_special_tokens=True).input_ids
        if len(ids) > self.max_label_len:
            ids = ids[: self.max_label_len - 1] + [self.tok.eos_token_id]
        # The decoder reads the prefix and the text; position k predicts token k+1. Loss on the text
        # and the final EOT only: the language, task and timestamp tokens are given, not predicted.
        dec_in = ids[:-1]
        labels = [-100] * (n_prefix - 1) + ids[n_prefix:]
        if self.lid_label and (r.get("text") or "").strip():
            labels[0] = ids[1]                          # position 0 predicts the language token
        return {"input_features": feats, "decoder_input_ids": dec_in, "labels": labels}


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
        maxlen = max(len(b["labels"]) for b in batch)
        pad = self.processor.tokenizer.pad_token_id
        dec, labels, attn = [], [], []
        for b in batch:
            assert len(b["decoder_input_ids"]) == len(b["labels"])
            n = maxlen - len(b["labels"])
            dec.append(b["decoder_input_ids"] + [pad] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["labels"]) + [0] * n)
        return {"input_features": feats,
                "decoder_input_ids": torch.tensor(dec, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "decoder_attention_mask": torch.tensor(attn, dtype=torch.long)}


def language_token_ids(tok):
    from transformers.models.whisper.tokenization_whisper import LANGUAGES
    ids = [tok.convert_tokens_to_ids(f"<|{c}|>") for c in LANGUAGES]
    return [i for i in ids if i is not None and i != tok.unk_token_id]


def lid_trainer_class():
    from transformers import Seq2SeqTrainer

    class LidTrainer(Seq2SeqTrainer):
        """Seq2SeqTrainer + KL(base || model) over the language tokens at position 0 (--lid-loss distill)."""

        def __init__(self, *a, lid_ids=None, lid_weight=1.0, base_model=None, **kw):
            super().__init__(*a, **kw)
            self.lid_ids = torch.tensor(lid_ids)
            self.lid_weight = lid_weight
            self.base_model = base_model                # None: the PEFT model with its adapter off
            self.last_lid_kl = 0.0

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch.nn.functional as F
            out = model(**inputs)
            ids = self.lid_ids.to(out.logits.device)
            student = out.logits[:, 0, :].index_select(-1, ids).float()
            first = {"input_features": inputs["input_features"],
                     "decoder_input_ids": inputs["decoder_input_ids"][:, :1]}
            with torch.no_grad():
                if self.base_model is not None:         # outside the trained model's autocast: match its dtype
                    if self.base_model.device != out.logits.device:
                        self.base_model.to(out.logits.device)
                    base = self.base_model(input_features=first["input_features"].to(self.base_model.dtype),
                                           decoder_input_ids=first["decoder_input_ids"]).logits
                else:
                    with model.disable_adapter():
                        base = model(**first).logits
                base = base[:, 0, :].index_select(-1, ids).float()
            kl = F.kl_div(F.log_softmax(student, -1), F.log_softmax(base, -1), log_target=True,
                          reduction="batchmean")
            self.last_lid_kl = float(kl.detach())
            loss = out.loss + self.lid_weight * kl
            return (loss, out) if return_outputs else loss

        def log(self, logs, *a, **kw):
            logs = dict(logs, lid_kl=round(self.last_lid_kl, 5))
            return super().log(logs, *a, **kw)

    return LidTrainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=Path("corpus/val.jsonl"))
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=["lora", "full"], default="lora")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                    help="attention projections AND the MLP. fc1/fc2 hold most of the "
                         "parameters in a Whisper block, so attention-only adapters leave the "
                         "bulk of each layer untouched.")
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 2e-4 lora, 1e-5 full. 1e-3 diverges on this data -- the "
                         "corpus mix hit loss 9.76 there and its checkpoint hallucinates on "
                         "100%% of silence.")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=0, help="0 = only at the end")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lid-loss", choices=["none", "label", "distill"], default="none",
                    help="what trains the language-token prediction (see the top of this file)")
    ap.add_argument("--lid-weight", type=float, default=1.0, help="weight of the distill KL term")
    ap.add_argument("--blank-lang-map", type=Path, default=None,
                    help="JSON {audio path: language base detects} for blank clips (asr/scripts/blank_langs.py)")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT",
                                                              "whisper-hallucination"))
    ap.add_argument("--wandb-group", default=None,
                    help="default: the mix name. Grouping by mix is what makes the "
                         "with-synth / without-synth pair readable in one view.")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases (off unless asked: user, 2026-10-23)")
    ap.add_argument("--no-wandb", action="store_true", help="kept for old command lines; W&B is off by default")
    args = ap.parse_args()

    # W&B only when asked for (--wandb) and there is a key; a sweep must not die because a token is missing.
    use_wandb = args.wandb and not args.no_wandb and bool(os.environ.get("WANDB_API_KEY"))
    wb = None

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    rows = [json.loads(l) for l in args.mix.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    random.Random(args.seed).shuffle(rows)
    blanks = sum(1 for r in rows if not (r.get("text") or "").strip())
    print(f"[train] {args.mix.name}: {len(rows)} clips, {blanks/len(rows):.0%} blank, "
          f"method={args.method}", flush=True)

    if use_wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or args.mix.stem,
            job_type=args.method,
            name=args.out.name,
            config={"mix": args.mix.stem, "method": args.method,
                    "lora_r": args.lora_r if args.method == "lora" else None,
                    "lora_alpha": args.lora_alpha if args.method == "lora" else None,
                    "lora_target_modules": (list(args.lora_target_modules)
                                            if args.method == "lora" else None),
                    "steps": args.steps, "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.batch_size * args.grad_accum,
                    "base_model": args.model, "n_clips": len(rows),
                    "blank_share": round(blanks / len(rows), 4),
                    "has_lexicon_synth": args.mix.stem in ("all", "plus_synth",
                                                           "balanced", "synth_heavy")},
            tags=[args.mix.stem, args.method], reinit=True)

    proc = WhisperProcessor.from_pretrained(args.model)
    # A full fine-tune keeps fp32 master weights (bf16 compute through autocast): in pure bf16 an
    # AdamW step at 1e-5 is ~0.05% of a typical weight and bf16 resolves ~0.4%, so most updates round
    # to nothing. LoRA leaves the base frozen in bf16; PEFT keeps the adapters themselves in fp32.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32 if args.method == "full" else torch.bfloat16)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_dropout, bias="none",
                              target_modules=list(args.lora_target_modules))
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    else:
        # non-reentrant: with fp32 master weights under bf16 autocast, the reentrant variant recomputes a
        # layer against autocast's cached weight casts, whose graph the first backward already freed
        # ("Trying to backward through the graph a second time")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lr = args.lr if args.lr is not None else (2e-4 if args.method == "lora" else 1e-5)
    if wb is not None:
        wb.config.update({"lr": lr}, allow_val_change=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=args.warmup,
        max_steps=args.steps,
        bf16=True,
        logging_steps=25,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or args.steps,
        report_to=["wandb"] if use_wandb else [],
        run_name=args.out.name,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
    )
    blank_map = json.loads(args.blank_lang_map.read_text()) if args.blank_lang_map else None
    ds = ClipDataset(rows, proc, lid_label=args.lid_loss == "label", blank_lang_map=blank_map)
    if args.lid_loss == "distill":
        base = None
        if args.method == "full":                       # a frozen copy; LoRA switches its adapter off
            base = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
            base = base.eval().requires_grad_(False)   # moved to the GPU on the first step (compute_loss)
        trainer = lid_trainer_class()(
            model=model, args=targs, train_dataset=ds, data_collator=Collator(proc),
            lid_ids=language_token_ids(proc.tokenizer), lid_weight=args.lid_weight, base_model=base)
    else:
        trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(proc))
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    # training cleared forced_decoder_ids (the dataset gives the prefix itself); the checkpoint must not keep
    # that, or generate() without a language leaves the TASK token to the model, which then sometimes
    # translates. The base model's generation config (which forces <|transcribe|>) is what ships.
    from transformers import GenerationConfig
    base_gen = GenerationConfig.from_pretrained(args.model)
    if args.method == "lora":
        model.save_pretrained(str(args.out / "adapter"))
        # Merge so the benchmark harness can load it like any other checkpoint.
        merged = model.merge_and_unload()
        merged.generation_config = base_gen
        merged.save_pretrained(str(args.out / "merged"), safe_serialization=True)
        proc.save_pretrained(str(args.out / "merged"))
    else:
        model.generation_config = base_gen
        model.save_pretrained(str(args.out / "merged"), safe_serialization=True)
        proc.save_pretrained(str(args.out / "merged"))
    (args.out / "run.json").write_text(json.dumps({
        "mix": str(args.mix), "method": args.method, "lr": lr, "steps": args.steps,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "lora_r": args.lora_r if args.method == "lora" else None,
        "lora_target_modules": (list(args.lora_target_modules)
                                if args.method == "lora" else None),
        "lora_alpha": args.lora_alpha if args.method == "lora" else None,
        "prefix": "fed",                      # decoder_input_ids carry the language token (see top)
        "lid_loss": args.lid_loss, "lid_weight": args.lid_weight if args.lid_loss == "distill" else None,
        "blank_lang": "base-detected" if args.blank_lang_map else "drawn from the mix",
        "master_weights": "fp32" if args.method == "full" else "bf16 base, fp32 adapters",
        "blank_langs": len(trainer.train_dataset.blank_langs),
        "n_clips": len(rows), "blank_share": round(blanks / len(rows), 4),
        "base_model": args.model,
        # The scorer reopens this run to attach the benchmark numbers, so training curve and
        # final metrics live on one W&B run instead of two unlinked halves.
        "wandb_run_id": wb.id if wb is not None else None,
        "wandb_project": args.wandb_project if wb is not None else None,
    }, indent=2))
    if wb is not None:
        wb.finish()
    print(f"[train] saved -> {args.out}/merged", flush=True)


if __name__ == "__main__":
    main()
