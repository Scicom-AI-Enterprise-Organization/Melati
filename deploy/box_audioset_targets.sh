#!/bin/bash
# The teachers' probabilities and the 16 kHz cache for AudioSet's training sets (audioset/targets.py): the balanced set
# first, then the unbalanced one, one slice a GPU, each following the fetch until its shards are all done.
#
#   bash deploy/box_audioset_targets.sh 0 1 2 4 5
#   SPLIT=strong_train SRC=$MELATI_DATA/audioset/strong/data/train EXPECT=3405 bash ...   (the strong subset)
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf
V=$MELATI_ROOT/.venv-sed/bin/python
GPUS=("${@:-0 1 2 4 5}")
n=${#GPUS[@]}
L=$MELATI_LOGS
SPLIT=${SPLIT:-unbal_train}                       # strong_train: the strongly labelled subset (SRC, EXPECT)
if [ "$SPLIT" = unbal_train ]; then
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $V audioset/targets.py --split bal_train > $L/astargets_bal.log 2>&1
    tail -2 $L/astargets_bal.log
fi
for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $V audioset/targets.py --split $SPLIT ${SRC:+--src $SRC} ${EXPECT:+--expect $EXPECT} \
        --slice $i/$n --follow > $L/astargets_${SPLIT}_$i.log 2>&1 &
done
wait
grep -h "^{" $L/astargets_${SPLIT}_*.log
echo ASTARGETS_DONE
