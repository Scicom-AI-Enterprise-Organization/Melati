#!/bin/bash
# Build, smoke-test and publish the released Whisper to Hugging Face, then its model card.
#
#   bash deploy/box_hf_release.sh <gpu>
#
# Weights from the merge (asr/train/merge_models.py -> merges/tts_full), the generation config that suppresses
# nothing and states the task (asr/release/build_release.py), the card from the summaries (asr/release/card.py,
# needs matplotlib, so .venv-melati). Each step's own log is kept; a failure stops the rest.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:-0}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=$GPU
REPO=Scicom-intl/Melati-Whisper-Large-V3
OUT=$MELATI_DATA/release/Melati-Whisper-Large-V3
R=$MELATI_DATA/results/asr
LOG=$MELATI_LOGS
$MELATI_ROOT/.venv-asr/bin/python asr/release/build_release.py --run $MELATI_DATA/asr/merges/tts_full --out $OUT \
    --arms-dir $MELATI_DATA/asr/arms_npz --repo $REPO --push > $LOG/hf_weights.log 2>&1
grep -E "pushed" $LOG/hf_weights.log
$MELATI_ROOT/.venv-melati/bin/python asr/release/card.py --run tts_full --repo $REPO --out $OUT \
    --summary $R/summary_merges.json --base-summary $R/summary.json --lid $R/lid.json \
    --run-json $MELATI_DATA/asr/merges/tts_full/run.json --push > $LOG/hf_card.log 2>&1
grep -E "pushed" $LOG/hf_card.log
echo HF_RELEASE_DONE
