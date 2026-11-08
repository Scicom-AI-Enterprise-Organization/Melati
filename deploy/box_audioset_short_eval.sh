#!/bin/bash
# audioset/short_eval.py: AudioSet's strongly labelled test clips cut to a podcast segment's lengths, the ensemble and
# v4 with the window filled three ways; one share of the clips a GPU, then the scores
# ($MELATI_DATA/results/audioset/short_eval.json). GPUS (default 0 1 2 4 5).
#
#   bash deploy/box_audioset_short_eval.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0 OMP_NUM_THREADS=8
GPUS=(${GPUS:-0 1 2 4 5})
n=${#GPUS[@]}
L=$MELATI_DATA/tmp
pids=()
for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $MELATI_ROOT/.venv-sed/bin/python -u audioset/short_eval.py run --slice $i/$n \
        > $L/short_eval_$i.log 2>&1 &
    pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
[ "$failed" = 0 ] || { echo "SHORT_FAILED: $failed of $n slices"; tail -3 $L/short_eval_*.log; exit 1; }
$MELATI_ROOT/.venv-sed/bin/python -u audioset/short_eval.py score --out $MELATI_DATA/results/audioset/short_eval.json
echo SHORT_EVAL_DONE
