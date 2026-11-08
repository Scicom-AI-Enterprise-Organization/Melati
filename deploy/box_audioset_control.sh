#!/bin/bash
# The control on Melati-AudioSet-EAT-Large v3's gain over v2. v3 is the clip half of frames1: v2 trained 5 more epochs
# on pool clean, without mixup, beside a frame head learning the strongly labelled clips' timed events. This runs the
# same thing with the frame loss off (audioset/frames_train.py --frame-weight 0), so only the timed events differ, then
# scores v2, v3 and the control in fp32 on the evaluation set (audioset/evaluate.py). Each run is taken at its last
# step, which for frames1 is also its best.
#
#   bash deploy/box_audioset_control.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf OMP_NUM_THREADS=4
V=$MELATI_ROOT/.venv-sed/bin
M=$MELATI_DATA/models/audioset
RUN=${RUN:-frames1_fw0}
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,4,5} $V/torchrun --nproc_per_node $(echo ${GPUS:-0,1,2,4,5} | tr , ' ' | wc -w) \
    --master_port ${PORT:-29514} audioset/frames_train.py --pool clean --init melati:$M/big_from_v1/best.pt \
    --out $M/$RUN --epochs 5 --frame-weight 0
$V/python audioset/clip_of_frames.py $M/$RUN/last.pt
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3} $V/python audioset/evaluate.py --fp32 \
    --models melati:$M/big_from_v1/best.pt melati:$M/frames1/clip.pt melati:$M/$RUN/clip.pt \
    --out $MELATI_DATA/results/audioset/student_v3_control_fp32.json
echo ASCONTROL_DONE
