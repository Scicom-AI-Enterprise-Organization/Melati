#!/bin/bash
# The frame-level head on Melati's AudioSet tagger (audioset/frames_train.py), one process a GPU, then the frame models
# against PretrainedSED's on the strongly labelled test clips (audioset/frames_eval.py).
#
#   INIT=$MELATI_DATA/models/audioset/big_from_v1/best.pt RUN=frames1 bash deploy/box_audioset_frames.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf OMP_NUM_THREADS=4
V=$MELATI_ROOT/.venv-sed/bin
O=$MELATI_DATA/models/audioset/$RUN
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,4,5} $V/torchrun --nproc_per_node $(echo ${GPUS:-0,1,2,4,5} | tr , ' ' | wc -w) \
    --master_port ${PORT:-29512} audioset/frames_train.py --pool ${POOL:-clean} --init melati:$INIT --out $O \
    --epochs ${EPOCHS:-5} ${ARGS:-}
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3} $V/python audioset/frames_eval.py --frames $O/best.pt \
    --out $MELATI_DATA/results/audioset/frames_eval_$RUN.json
echo ASFRAMES_DONE
