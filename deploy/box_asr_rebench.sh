#!/bin/bash
# Re-score trained checkpoints after the task-token fix, on one GPU, one after another.
#
#   bash deploy/box_asr_rebench.sh <gpu> <run dir> [...]
#
# Checkpoints trained before the fix were saved with forced_decoder_ids cleared, so generate() without a
# language left the TASK token to the model, and the fine-tunes (which obey their tokens) sometimes
# translated: every number with the language detected was measured that way, the paper's own recipe's
# FLEURS loss included. Each checkpoint first gets the base model's generation config (which forces
# <|transcribe|>, as vLLM's prompt always does), then the whole benchmark again (asr/train/eval_run.sh,
# which now also states task="transcribe"). A run may wait for its own training or first benchmark.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}; shift
BASE=$(ls -d $MELATI_DATA/hf/hub/models--openai--whisper-large-v3/snapshots/*/ | head -1)
for RUN in "$@"; do
  until [ -d "$RUN/merged" ] && ! pgrep -f "run_benchmark.py --model $RUN/merged" > /dev/null; do sleep 60; done
  cp "$BASE/generation_config.json" "$RUN/merged/generation_config.json"
  echo "=== $(basename "$RUN") on gpu $GPU $(date +%T)"
  bash $MELATI_REPO/asr/train/eval_run.sh "$RUN" "$GPU" 2>&1 | tail -3
done
echo REBENCH_DONE
