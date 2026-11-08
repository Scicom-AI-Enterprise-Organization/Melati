#!/bin/bash
# The first ASR sweep: corpus (once every arm exists), mixes, then one job per GPU.
#
#   bash deploy/box_asr_train.sh
#
# Jobs (LoRA r64, alpha 128, lr 5e-4, effective batch 16: the best configuration of
# "How to Reduce Whisper Hallucination"; only the mix and the length vary):
#   GPU0  all            1000 steps   the paper's recipe, rebuilt here: the no-replay control
#   GPU1  all_cv22_r25   1000 steps   + Common Voice 22, 25% of the clips
#   GPU2  all_cv22_r50   1000 steps   + Common Voice 22, 50% of the clips
#   GPU3  all            3000 steps   is the paper's FLEURS loss a transient of a short run?
#   GPU4  all_cv22_r50   3000 steps   replay, run longer
# Every job is benchmarked right after training (asr/train/eval_run.sh).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
until [ -f "$D/logs/emilia.done" ]; do sleep 30; done
echo "=== corpus (with the Malaysian podcast positives) $(date +%T)"
rm -rf "$D/corpus" "$D/logs/corpus.done"
$PY scripts/build_training_corpus.py --audio-root "$D/audio_train" --out "$D/corpus" \
    --nonspeech-hours 105 --positive-hours 60 2>&1 | tee "$D/logs/corpus.log" | tail -14
touch "$D/logs/corpus.done"
echo "=== mixes $(date +%T)"
$PY train/build_mix.py --corpus "$D/corpus/train.jsonl" --cv22 "$D/cv22/train/manifest.csv" \
    --cache "$D/hub" --out "$D/mixes" 2>&1 | tail -12
mkdir -p "$D/logs"
bash train/grid.sh 0 all:64:5e-4:1000          > "$D/logs/grid_0.log" 2>&1 &
bash train/grid.sh 1 all_cv22_r25:64:5e-4:1000 > "$D/logs/grid_1.log" 2>&1 &
bash train/grid.sh 2 all_cv22_r50:64:5e-4:1000 > "$D/logs/grid_2.log" 2>&1 &
bash train/grid.sh 3 all:64:5e-4:3000          > "$D/logs/grid_3.log" 2>&1 &
bash train/grid.sh 4 all_cv22_r50:64:5e-4:3000 > "$D/logs/grid_4.log" 2>&1 &
wait
echo ASR_SWEEP_DONE
