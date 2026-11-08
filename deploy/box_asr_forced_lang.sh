#!/bin/bash
# FLEURS decoded with each clip's own language token, for base and the first fine-tunes: does a
# fine-tune still TRANSCRIBE every language, with language identification taken out of the question?
# Writes bench/<model>/fleurs_forced.jsonl beside the auto-detected run (every row records `lid`).
#
#   bash deploy/box_asr_forced_lang.sh <gpu>
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
for spec in "openai/whisper-large-v3:$D/runs/base_large_v3/bench" \
            "$D/runs/v3_lora_r64_all_lr5e-4_s1000/merged:$D/runs/v3_lora_r64_all_lr5e-4_s1000/bench" \
            "$D/runs2/v3_lora_r64_all_lr5e-4_s1000/merged:$D/runs2/v3_lora_r64_all_lr5e-4_s1000/bench" \
            "$D/runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000/merged:$D/runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000/bench"; do
  model=${spec%%:*}; out=${spec#*:}
  echo "=== $model $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU $PY bench/run_benchmark.py --model "$model" --device cuda:0 --batch-size 24 \
      --arms fleurs --force-language --pad-tag _forced --overwrite \
      --fleurs-cache "$D/bench/fleurs_sample.parquet" --out "$out" 2>&1 | grep -E "^\[done\]|Error|Traceback" | tail -3
done
echo FORCED_LANG_DONE
