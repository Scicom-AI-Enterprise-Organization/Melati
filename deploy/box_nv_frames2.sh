#!/bin/bash
# The frame-level laugh detector, v2: AMI plus NonverbalTTS's training clips (laugh-free clips as negatives in every
# frame, each laugh tag a multiple-instance bag around its word boundary), scored like v1 beside the gap head named.
#
#   bash deploy/box_nv_frames2.sh 3 rel5
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-3}
GAP=${2:-rel5}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
$M nonverbal/frames.py extract > $LOG/nvframes2_extract.log 2>&1
grep -E "nvtts|EXTRACT_DONE" $LOG/nvframes2_extract.log
$M nonverbal/frames.py train --nvtts --out $MD/nv_frames2/whisper.pt > $LOG/nvframes2_train.log 2>&1
grep -E "rest|epoch|TRAIN_DONE" $LOG/nvframes2_train.log
while [ ! -f $R/gap_eval_$GAP.json ]; do sleep 60; done
$M scripts/frames_diag.py --model $MD/nv_frames2/whisper.pt 2>&1 | grep -E "^calib|^test|t=0.(5|8|9)"
$M nonverbal/frames.py eval --model $MD/nv_frames2/whisper.pt --gap-heads $MD/nv_heads_$GAP \
    --gap-thresholds $R/gap_eval_$GAP.json --out $R/frames2_eval_$GAP.json > $LOG/nvframes2_eval.log 2>&1
grep -E "^AMI|^NonverbalTTS|FRAMES_EVAL_DONE" $LOG/nvframes2_eval.log
echo NV_FRAMES2_DONE
