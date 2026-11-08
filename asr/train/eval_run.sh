#!/bin/bash
# Score one checkpoint on the whole benchmark, then fold it into eval.json.
#
#   bash asr/train/eval_run.sh $MELATI_DATA/asr/runs/<run> <gpu>
#   bash asr/train/eval_run.sh BASE <gpu>          # the untuned openai/whisper-large-v3
#
# Arms: silence music nonspeech (did hallucination fall), reduplication (did looping fall, or
# did it start deleting), speech_in_noise, genuine and genuine_isolated (Malaysian speech),
# librispeech_test_clean (English), fleurs (58 languages, never trained on: the honest
# multilingual guard), cv22 (in-domain for the replay mixes), lexicon_synth test (can it still
# say the phrases it used to invent), wild test (real audio).
#
# --overwrite is not optional: run_benchmark skips an arm whose jsonl exists, so a re-trained
# checkpoint in the same directory would otherwise be scored on the old transcripts.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../../deploy/env.sh"
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled   # no W&B (user, 2026-10-23)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # several benchmarks share a GPU; fragmentation OOMs them
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
RUN=${1:?usage: eval_run.sh <run dir | BASE> [gpu]}
GPU=${2:-0}
if [ "$RUN" = "BASE" ]; then
  MODEL=openai/whisper-large-v3; RUN=$D/runs/base_large_v3; mkdir -p "$RUN"
else
  MODEL="$RUN/merged"
  [ -d "$MODEL" ] || { echo "no merged model at $MODEL"; exit 1; }
fi

CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL $PY bench/run_benchmark.py \
    --model "$MODEL" --device cuda:0 --batch-size 48 --overwrite \
    --fleurs-cache "$D/bench/fleurs_sample.parquet" --cv22-test "$D/cv22_test50/test/manifest.csv" \
    --arms silence music nonspeech reduplication speech_in_noise genuine genuine_isolated \
           librispeech_test_clean fleurs cv22 lexicon_synth wild \
    --out "$RUN/bench" > "$RUN/bench.log" 2>&1
rc=$?
grep -E "^\[done\]|Error|Traceback" "$RUN/bench.log" | tail -14
# a crashed benchmark leaves the PREVIOUS transcripts in place (an arm is written to .part and renamed only when
# it completes), so scoring after a crash would publish old numbers under a new date
[ $rc -eq 0 ] || { echo "BENCH FAILED ($rc): not scoring $RUN"; exit 1; }
# FLEURS again with each clip's own language token: transcription measured apart from language ID
# (a fine-tune that keeps transcribing but mis-identifies the language loses FLEURS for the second reason)
CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL $PY bench/run_benchmark.py \
    --model "$MODEL" --device cuda:0 --batch-size 48 --overwrite --arms fleurs --force-language --pad-tag _forced \
    --fleurs-cache "$D/bench/fleurs_sample.parquet" --out "$RUN/bench" >> "$RUN/bench.log" 2>&1 \
  || { echo "BENCH FAILED (forced FLEURS): not scoring $RUN"; exit 1; }

$PY bench/score_eval_run.py --run "$RUN" --lexicon "$D/lexicon/combined_lexicon.csv" 2>&1 | tail -30
