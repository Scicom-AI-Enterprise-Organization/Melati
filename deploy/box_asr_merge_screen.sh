#!/bin/bash
# Build one merged model (asr/train/merge_models.py) and screen it on the arms that separate the fine-tunes:
# the voice-free arms, reduplication (loops), LibriSpeech, Malaysian speech and FLEURS. A merge worth keeping
# then gets the whole benchmark (asr/train/eval_run.sh) like any other run.
#
#   bash deploy/box_asr_merge_screen.sh <gpu> <name> <method> [k=v ...] -- <run> [<run> ...]
#   k=v: lam=1.0 density=0.2 weights="1 1" check=1
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}; NAME=${2:?name}; METHOD=${3:?method}; shift 3
EXTRA=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case $1 in
    lam=*) EXTRA+=(--lam "${1#lam=}");; density=*) EXTRA+=(--density "${1#density=}");;
    weights=*) EXTRA+=(--weights ${1#weights=});; check=1) EXTRA+=(--check);;
  esac; shift
done
shift
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
OUT=$D/merges/$NAME
mkdir -p "$OUT"
if [ ! -s "$OUT/merged/model.safetensors" ]; then
  $PY train/merge_models.py --method "$METHOD" "${EXTRA[@]}" --out "$OUT" --runs "$@" > "$OUT/merge.log" 2>&1 \
    || { echo "MERGE FAILED $NAME"; tail -5 "$OUT/merge.log"; exit 1; }
fi
echo "=== $NAME merged $(date +%T)"
CUDA_VISIBLE_DEVICES=$GPU $PY bench/run_benchmark.py --model "$OUT/merged" --device cuda:0 --batch-size 48 --overwrite \
    --fleurs-cache "$D/bench/fleurs_sample.parquet" \
    --arms silence music nonspeech reduplication librispeech_test_clean genuine fleurs \
    --out "$OUT/bench" > "$OUT/bench.log" 2>&1 || { echo "BENCH FAILED $NAME"; exit 1; }
$PY bench/score_eval_run.py --run "$OUT" --lexicon "$D/lexicon/combined_lexicon.csv" > /dev/null 2>&1
echo "SCREEN_DONE $NAME $(date +%T)"
