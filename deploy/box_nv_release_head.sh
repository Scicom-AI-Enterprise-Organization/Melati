#!/bin/bash
# The non-verbal head that is released: PodcastFillers (events and candidates), NonverbalTTS candidates and COUGHVID
# coughs in place of VocalSound (whose ShareAlike cannot sit beside PodcastFillers' non-commercial terms), scored with
# the shipped gap rule on NonverbalTTS like every head before it.
#
#   bash deploy/box_nv_release_head.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
H=$MELATI_DATA/models/nv_heads_rel
LOG=$MELATI_LOGS
mkdir -p $H
[ -s $NV/manifest_coughvid.jsonl ] || $M nonverbal/build_manifest.py --coughvid $NV/raw/coughvid/public_dataset_v3.zip \
    --out $NV --name manifest_coughvid > $LOG/nvrel_manifest.log 2>&1
tail -2 $LOG/nvrel_manifest.log
$M emotion/extract.py --manifest $NV/manifest_coughvid.jsonl --encoder whisper --out $NV/feats > $LOG/nvrel_extract.log 2>&1
grep -E "EXTRACT_DONE" $LOG/nvrel_extract.log
cat $NV/manifest_v3_novs.jsonl $NV/manifest_coughvid.jsonl > $NV/manifest_rel.jsonl
$M nonverbal/train.py --manifest $NV/manifest_rel.jsonl --feats $NV/feats --encoder whisper \
    --out $R/head_whisper_rel.json --save-model $H/whisper.pt > $LOG/nvrel_train.log 2>&1
grep -E "test all|TRAIN_DONE" $LOG/nvrel_train.log
MELATI_NV_HEADS=$H $M nonverbal/gap_rule.py eval --model whisper --rules '{"floor_db": 2}' \
    --out $R/gap_eval_rel.json > $LOG/nvrel_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nvrel_eval.log | tail -2
echo NV_REL_DONE
