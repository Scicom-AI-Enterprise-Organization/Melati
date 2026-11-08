#!/bin/bash
# A few steps of every trainer mode on a slice of a mix: does each one run, and log what it should?
#
#   bash deploy/box_asr_smoke.sh <gpu>
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
# SMOKE="full:distill" runs one; the default runs all three
for spec in ${SMOKE:-lora:label lora:distill full:distill}; do
  m=${spec%%:*}; lid=${spec#*:}
  out=$MELATI_DATA/tmp/smoke_${m}_${lid}; rm -rf "$out"; mkdir -p $MELATI_DATA/tmp
  extra="--lora-r 8 --lora-alpha 16"; [ "$m" = full ] && extra=""
  echo "=== $m $lid $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU $PY train/finetune_whisper.py --mix "$D/mixes/all_cv22_r25.jsonl" --limit 64 \
      --method "$m" --lid-loss "$lid" --steps 3 --batch-size 2 --grad-accum 1 --warmup 0 --no-wandb \
      --out "$out" $extra > "$out.log" 2>&1
  grep -E "lid_kl|train_loss|saved" "$out.log" | tail -3; grep -A40 Traceback "$out.log" | grep -vE '^\s*\^+\s*$' | tail -30
  test -s "$out/run.json" && grep -o '"lid_loss": "[a-z]*"' "$out/run.json"
done
rm -rf $MELATI_DATA/tmp/smoke_*/
echo SMOKE_DONE
