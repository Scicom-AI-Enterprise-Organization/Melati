#!/bin/bash
# One or more Melati ASR jobs on ONE gpu. A job is `mix:rank:lr[:steps[:targets[:lid]]]`; rank 0 =
# full fine-tune; targets `attnmlp` (default: q/k/v/out_proj + fc1/fc2) or `attn` (projections only);
# lid = finetune_whisper.py --lid-loss (`none` default, `label`, `distill`; it suffixes the run name).
# BLANK_LANG_MAP=<json> trains blank clips under base's detected language (suffix _blbase).
#
#   bash asr/train/grid.sh 0 all:64:5e-4 all_cv22_r50:64:5e-4:3000 all:64:5e-4:1000:attn all:64:5e-4:1000:attnmlp:distill
#
# Runs land in $RUNS (default $MELATI_DATA/asr/runs2). runs/ holds the first sweep, trained
# before the decoder was fed its prefix (finetune_whisper.py, top): kept as the record of that bug.
#
# Each job trains, then is benchmarked on every arm (asr/train/eval_run.sh), so a GPU never
# idles waiting for the rest of the grid. A job whose run has eval.json and a merged model is
# skipped, which makes the grid resumable; FORCE=1 re-runs it anyway.
#
# The defaults are the best configuration of "How to Reduce Whisper Hallucination": LoRA on
# q/k/v/out_proj + fc1/fc2, alpha = 2r, effective batch 16, warmup 50, 1000 steps.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../../deploy/env.sh"
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled   # no W&B (user, 2026-10-23)
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
GPU=${1:?usage: grid.sh <gpu> <mix:rank:lr[:steps]> [...]}
shift
FORCE=${FORCE:-0}

RUNS=${RUNS:-$D/runs2}
mkdir -p "$RUNS" "$D/logs"
[ -e "$RUNS/base_large_v3" ] || ln -s "$D/runs/base_large_v3" "$RUNS/base_large_v3"
for job in "$@"; do
  IFS=: read -r mix r lr steps tgt lid <<< "$job"
  steps=${steps:-1000}; tgt=${tgt:-attnmlp}; lid=${lid:-none}
  sfx=""; [ "$lid" != none ] && sfx="_lid$lid"
  # BLANK_LANG_MAP: blank clips trained under the language base detects on them (asr/scripts/blank_langs.py)
  [ -n "${BLANK_LANG_MAP:-}" ] && sfx="${sfx}_blbase"
  if [ "$r" = "0" ]; then
    name="v3_full_${mix}_lr${lr}_s${steps}${sfx}"; method=full; extra="--batch-size 4 --grad-accum 4"
  else
    if [ "$tgt" = "attn" ]; then fam=loraattn; mods="q_proj k_proj v_proj out_proj"
    else fam=lora; mods="q_proj k_proj v_proj out_proj fc1 fc2"; fi
    name="v3_${fam}_r${r}_${mix}_lr${lr}_s${steps}${sfx}"; method=lora
    extra="--lora-r $r --lora-alpha $((r*2)) --batch-size 8 --grad-accum 2 --lora-target-modules $mods"
  fi
  RUN=$RUNS/$name
  if [ "$FORCE" != "1" ] && [ -s "$RUN/eval.json" ] && [ -d "$RUN/merged" ]; then
    echo "[skip] $name"; continue
  fi
  echo "=== $name (gpu $GPU) $(date +%T)"
  if [ "$FORCE" = "1" ] || [ ! -d "$RUN/merged" ]; then
    CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL $PY train/finetune_whisper.py \
        --mix "$D/mixes/$mix.jsonl" --method $method --lr "$lr" --steps "$steps" \
        --out "$RUN" --lid-loss "$lid" ${BLANK_LANG_MAP:+--blank-lang-map "$BLANK_LANG_MAP"} $extra > "$D/logs/train2_$name.log" 2>&1
    grep -E "trainable params|train_loss|saved|CUDA out|Error" "$D/logs/train2_$name.log" | tail -4
  fi
  bash train/eval_run.sh "$RUN" "$GPU"
  echo "=== done $name $(date +%T)"
done
