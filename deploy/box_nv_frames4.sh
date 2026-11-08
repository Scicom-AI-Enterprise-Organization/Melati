#!/bin/bash
# The frame-level detector, v4: v3 (AMI, PodcastFillers' CC BY episodes) plus COUGHVID's and Coswara's coughs packed
# into 30 s windows. v3 learnt coughs from AMI's meetings alone (PodcastFillers labels none) and never fired on
# NonverbalTTS's. Scored like v3, and on NonverbalTTS at every threshold.
#
#   bash deploy/box_nv_frames4.sh 1
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-1}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
[ -s $NV/manifest_coughvid_f.jsonl ] || $M nonverbal/build_manifest.py --coughvid $NV/raw/coughvid/public_dataset_v3.zip \
    --out $NV --name manifest_coughvid_f > $LOG/nvframes4_coughvid.log 2>&1
tail -1 $LOG/nvframes4_coughvid.log
$M nonverbal/frames.py extract --coughs $NV/manifest_coughvid_f.jsonl $NV/manifest_coswara.jsonl \
    > $LOG/nvframes4_extract.log 2>&1
grep -E "coughs|EXTRACT_DONE" $LOG/nvframes4_extract.log
$M nonverbal/frames.py train --pf --coughs --out $MD/nv_frames4/whisper.pt > $LOG/nvframes4_train.log 2>&1
grep -E "PodcastFillers|coughs|epoch|TRAIN_DONE" $LOG/nvframes4_train.log
$M nonverbal/frames.py eval --model $MD/nv_frames4/whisper.pt --gap-heads $MD/nv_heads_rel4 \
    --gap-thresholds $R/gap_eval_rel4pf.json --out $R/frames4_eval_pf.json > $LOG/nvframes4_eval.log 2>&1
grep -E "^AMI|^PodcastFillers|FRAMES_EVAL_DONE" $LOG/nvframes4_eval.log
$M scripts/frames_diag.py --model $MD/nv_frames4/whisper.pt --out $R/frames4_nvtts.json 2>&1 \
    | grep -E "^calib|^test|t=0.9:|t=0.95:|t=0.99:|t=0.999:"
echo NV_FRAMES4_DONE
