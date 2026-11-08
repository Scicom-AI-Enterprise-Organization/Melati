#!/bin/bash
# Where Melati's clip tagger hears each sound, read off short windows (audioset/window_eval.py): one process a GPU over
# a slice of AudioSet's strongly labelled test clips, then the merge, which scores the timing against PretrainedSED's.
#
#   bash deploy/box_audioset_windows.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf OMP_NUM_THREADS=4
PY=$MELATI_ROOT/.venv-sed/bin/python
CKPT=${CKPT:-$MELATI_DATA/models/audioset/frames1_fw0/clip.pt}
GPUS=${GPUS:-0,1,4,5}
n=$(echo $GPUS | tr , ' ' | wc -w)
i=0
pids=""
for g in $(echo $GPUS | tr , ' '); do
    CUDA_VISIBLE_DEVICES=$g $PY audioset/window_eval.py --ckpt $CKPT --shard $i/$n \
        > $MELATI_LOGS/aswin_$i.log 2>&1 &
    pids="$pids $!"
    i=$((i + 1))
done
for p in $pids; do
    wait $p
done
CUDA_VISIBLE_DEVICES= $PY audioset/window_eval.py --ckpt $CKPT --merge \
    --out $MELATI_DATA/results/audioset/window_eval.json
echo ASWINDOWS_DONE
