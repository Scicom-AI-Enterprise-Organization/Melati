#!/bin/bash
# Re-score checkpoints decoded as vLLM serves them (asr/bench/run_benchmark.py's default since the
# suppression check: no token forbidden, so a model can return an empty transcript), one after another
# on one GPU. A run already scored that way on every arm is skipped (asr/scripts/bench_protocol.py), so
# this is resumable, and a run still training or being benchmarked by another job is waited for.
#
#   bash deploy/box_asr_rescore.sh <gpu> <run dir|BASE> [...]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}; shift
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
BASE=$(ls -d $MELATI_DATA/hf/hub/models--openai--whisper-large-v3/snapshots/*/ | head -1)
for RUN in "$@"; do
  dir=$RUN; [ "$RUN" = BASE ] && dir=$D/runs/base_large_v3
  if [ "$RUN" != BASE ]; then
    until [ -d "$RUN/merged" ] && ! pgrep -f "run_benchmark.py --model $RUN/merged" > /dev/null; do sleep 60; done
    # checkpoints saved before the trainer kept it carry no forced task token (box_asr_rebench.sh)
    cp "$BASE/generation_config.json" "$RUN/merged/generation_config.json"
  fi
  if [ -z "$($PY $MELATI_REPO/asr/scripts/bench_protocol.py "$dir" --stale-only)" ]; then
    echo "[skip] $(basename "$dir") already scored as vLLM decodes"; continue
  fi
  echo "=== $(basename "$dir") on gpu $GPU $(date +%T)"
  bash $MELATI_REPO/asr/train/eval_run.sh "$RUN" "$GPU" > /tmp/rescore.$$.log 2>&1
  rc=$?; tail -3 /tmp/rescore.$$.log
  [ $rc -eq 0 ] || echo "RESCORE FAILED: $(basename "$dir")"
done
echo RESCORE_DONE
