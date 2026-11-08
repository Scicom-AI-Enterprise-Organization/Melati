#!/bin/bash
# Melati-AudioSet v5: more teachers. CED-base + EAT-large + SSLAM average 0.5166 mAP on the evaluation set; with our
# own v3 and v4 0.5188, with BEATs (iter3+ AS2M cpt2, 0.4835 alone) 0.5200, with both 0.5218, while PretrainedSED's
# taggers read at clip level only lower it (results/audioset/teachers_more_fp32.json, teachers_beats_fp32.json). So
# BEATs, v3 and v4 tag every training clip (audioset/targets_add.py, one process a GPU; a shard that has a teacher's
# file is skipped), the pool averages all six, and v5 is trained from v4 on it without mixup, then scored in fp32.
#
#   bash deploy/box_audioset_v5.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf OMP_NUM_THREADS=4
V=$MELATI_ROOT/.venv-sed/bin
M=$MELATI_DATA/models/audioset
GPUS=${GPUS:-0,1,2,4,5}
n=$(echo $GPUS | tr , ' ' | wc -w)
i=0
pids=""
for g in $(echo $GPUS | tr , ' '); do
    CUDA_VISIBLE_DEVICES=$g $V/python audioset/targets_add.py --sources strong_train bal_train unbal_train \
        --slice $i/$n --teacher v4=melati:$M/frames1_fw0/clip.pt --teacher v3=melati:$M/frames1/clip.pt \
        --teacher beats=beats \
        > $MELATI_LOGS/asv5_add_$i.log 2>&1 &
    pids="$pids $!"
    i=$((i + 1))
done
for p in $pids; do
    wait $p
done
POOL=big6 SOURCES="strong_train bal_train unbal_train" \
    POOL_ARGS="--index-only --teachers ced-base eat-large sslam beats v3 v4" RUN=v5_big6 EPOCHS=${EPOCHS:-3} PORT=29515 \
    ARGS="--init melati:$M/frames1_fw0/clip.pt --mixup 0 --eval-every 3000" bash deploy/box_audioset_train.sh
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3} $V/python audioset/evaluate.py --fp32 --models ced-base eat-large sslam beats \
    melati:$M/frames1_fw0/clip.pt melati:$M/v5_big6/best.pt --add-to ced-base,eat-large,sslam,beats \
    --out $MELATI_DATA/results/audioset/student_v5_fp32.json
echo ASV5_DONE
