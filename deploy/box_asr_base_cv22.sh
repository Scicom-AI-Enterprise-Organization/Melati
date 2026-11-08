#!/bin/bash
# Base whisper-large-v3 on the 50-per-language CV22 test set, once its full benchmark is done, so base and
# every fine-tune are compared on the same CV22 clips (the full run used the 20-per-language set).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1
D=$MELATI_DATA/asr
RUN=$D/runs/base_large_v3
until [ -s "$RUN/eval.json" ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=${1:-5} $MELATI_ROOT/.venv-asr/bin/python bench/run_benchmark.py --model openai/whisper-large-v3 \
    --device cuda:0 --batch-size 24 --overwrite --arms cv22 --cv22-test "$D/cv22_test50/test/manifest.csv" \
    --out "$RUN/bench" > "$RUN/bench_cv22_50.log" 2>&1
$MELATI_ROOT/.venv-asr/bin/python bench/score_eval_run.py --run "$RUN" --lexicon "$D/lexicon/combined_lexicon.csv" \
    --no-wandb 2>&1 | tail -5
echo BASE_CV22_50_DONE
