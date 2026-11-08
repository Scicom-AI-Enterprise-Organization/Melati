#!/bin/bash
# The frame-level detector, v3: AMI plus PodcastFillers' CC BY training episodes (hand-timed laughs and breaths, known
# only inside the regions its annotators labelled), the licence family of the PodcastFillers-trained gap head. Scored
# on AMI, PodcastFillers' test episodes and NonverbalTTS beside that head (rel4, thresholds chosen on PodcastFillers'
# validation episodes); v2 (AMI + NonverbalTTS) scored the same way for comparison.
#
#   bash deploy/box_nv_frames3.sh 1
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-1}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
GAP="--gap-heads $MD/nv_heads_rel4 --gap-thresholds $R/gap_eval_rel4pf.json"
$M nonverbal/frames.py extract --pf > $LOG/nvframes3_extract.log 2>&1        # nothing left to do if already run
grep -E "EXTRACT_DONE" $LOG/nvframes3_extract.log
$M nonverbal/frames.py train --pf --out $MD/nv_frames3/whisper.pt > $LOG/nvframes3_train.log 2>&1
grep -E "PodcastFillers|epoch|TRAIN_DONE" $LOG/nvframes3_train.log
for v in 3 2; do
    $M nonverbal/frames.py eval --model $MD/nv_frames$v/whisper.pt $GAP --out $R/frames${v}_eval_pf.json \
        > $LOG/nvframes${v}_evalpf.log 2>&1
    echo "== frames v$v"
    grep -E "^AMI|^PodcastFillers|^NonverbalTTS|FRAMES_EVAL_DONE" $LOG/nvframes${v}_evalpf.log
done
echo NV_FRAMES3_DONE
