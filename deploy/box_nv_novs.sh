#!/bin/bash
# The non-verbal head without VocalSound, for release. VocalSound is CC BY-SA 4.0 and PodcastFillers' licence allows
# derivatives for non-commercial research only, so weights trained on both have no licence that satisfies each; the
# released head keeps PodcastFillers (events and candidates) and NonverbalTTS (Apache-2.0) and goes out CC BY-NC 4.0.
#
#   bash deploy/box_nv_novs.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
H=$MELATI_DATA/models/nv_heads_novs
LOG=$MELATI_LOGS
mkdir -p $H
grep -v '"corpus": "vocalsound"' $NV/manifest_v3.jsonl > $NV/manifest_v3_novs.jsonl
wc -l $NV/manifest_v3.jsonl $NV/manifest_v3_novs.jsonl
$M nonverbal/train.py --manifest $NV/manifest_v3_novs.jsonl --feats $NV/feats --encoder whisper \
    --out $R/head_whisper_novs.json --save-model $H/whisper.pt > $LOG/nvnovs_train.log 2>&1
grep -E "test all|TRAIN_DONE" $LOG/nvnovs_train.log
MELATI_NV_HEADS=$H $M nonverbal/gap_rule.py eval --model whisper --rules '{"floor_db": 2}' \
    --out $R/gap_eval_novs.json > $LOG/nvnovs_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nvnovs_eval.log | tail -2
echo NV_NOVS_DONE
