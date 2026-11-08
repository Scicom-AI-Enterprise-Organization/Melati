#!/bin/bash
# The frame-level laugh detector (nonverbal/frames.py): AMI features and frame labels, training on AMI train, and
# scoring on AMI test and NonverbalTTS test, alone and beside the gap head it would ship with (rel3, from
# deploy/box_nv_rel3.sh, which this waits for).
#
#   bash deploy/box_nv_frames.sh 3
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-3}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
while [ ! -f $LOG/amifetch.done ]; do sleep 30; done
$M nonverbal/frames.py extract > $LOG/nvframes_extract.log 2>&1
grep -E "EXTRACT_DONE" $LOG/nvframes_extract.log
$M nonverbal/frames.py train --out $MD/nv_frames/whisper.pt > $LOG/nvframes_train.log 2>&1
grep -E "epoch|TRAIN_DONE" $LOG/nvframes_train.log
while [ ! -f $R/ami_eval_rel3.json ]; do sleep 60; done           # rel3 trained and scored
$M nonverbal/frames.py eval --model $MD/nv_frames/whisper.pt --gap-heads $MD/nv_heads_rel3 \
    --gap-thresholds $R/gap_eval_rel3.json --out $R/frames_eval.json > $LOG/nvframes_eval.log 2>&1
grep -E "^AMI|^NonverbalTTS|FRAMES_EVAL_DONE" $LOG/nvframes_eval.log
echo NV_FRAMES_DONE
