#!/bin/bash
# rel6: rel5 (NonverbalTTS, AMI, COUGHVID and Coswara; no PodcastFillers) with the coughs pool at a third of a corpus's
# share. As one pool of one class it took a whole corpus's share, so coughs were 45% of the loss (29% in rel2): rel5's
# cough tags on NonverbalTTS's calibration clips never reached 80% precision. Then everything the release reads:
# NonverbalTTS, AMI, PodcastFillers' test episodes through the pipeline, and the frame detector (v2) beside this head.
#
#   bash deploy/box_nv_rel6.sh 3
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-3}
V=${2:-rel6}
W=${3:-'{"coughs": 0.33}'}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'
mkdir -p $MD/nv_heads_$V
$M nonverbal/train.py --manifest $NV/manifest_rel5.jsonl --feats $NV/feats --encoder whisper --pool-weights "$W" \
    --out $R/head_whisper_$V.json --save-model $MD/nv_heads_$V/whisper.pt > $LOG/nv${V}_train.log 2>&1
grep -E "test all|TRAIN_DONE" $LOG/nv${V}_train.log
MELATI_NV_HEADS=$MD/nv_heads_$V $M nonverbal/gap_rule.py eval --model whisper --rules "$RULE" \
    --out $R/gap_eval_$V.json > $LOG/nv${V}_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nv${V}_eval.log | tail -1
$M scripts/gap_curve.py --heads $MD/nv_heads_$V --classes cough 2>&1 | grep -E "^calib|^test"
$M nonverbal/eval_ami.py --heads $MD/nv_heads_$V --thresholds $R/gap_eval_$V.json --out $R/ami_eval_$V.json \
    > $LOG/nv${V}_ami.log 2>&1
grep -vE "warn|Warn|^  " $LOG/nv${V}_ami.log | tail -3
bash deploy/box_nv_pf_eval.sh $V $G
$M nonverbal/frames.py eval --model $MD/nv_frames2/whisper.pt --gap-heads $MD/nv_heads_$V \
    --gap-thresholds $R/gap_eval_$V.json --out $R/frames2_eval_$V.json > $LOG/nvframes2_eval_$V.log 2>&1
grep -E "^AMI|^NonverbalTTS|FRAMES_EVAL_DONE" $LOG/nvframes2_eval_$V.log
echo NV_REL6_DONE
