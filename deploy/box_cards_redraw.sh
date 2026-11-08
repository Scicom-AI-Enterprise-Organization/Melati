#!/bin/bash
# Redraw every Hugging Face card's figures in the preprint's style (melati/plotstyle.py) without touching weights or
# data: each card's own generator, the models' into $MELATI_DATA/release/cards/<name> and the dataset's into its
# publish folder. Nothing is uploaded here; scripts/push_card_figures.py does that once the figures have been looked at.
#
#   bash deploy/box_cards_redraw.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
C=$MELATI_DATA/release/cards
R=$MELATI_DATA/results/asr
mkdir -p $C
$M asr/release/card.py --run tts_full --repo Scicom-intl/Melati-Whisper-Large-V3 --out $C/asr \
    --summary $R/summary_merges.json --base-summary $R/summary.json --lid $R/lid.json \
    --run-json $MELATI_DATA/asr/merges/tts_full/run.json > $C/asr.log 2>&1
$M emotion/release.py --figures-only --head /dev/null --out $C/emotion \
    --repo Scicom-intl/Melati-Emotion-Whisper-Large-V3 > $C/emotion.log 2>&1
$M nonverbal/release.py --figures-only --out $C/nonverbal \
    --repo Scicom-intl/Melati-NonVerbal-Whisper-Large-V3 > $C/nonverbal.log 2>&1
$M scripts/dataset_card.py --hf $MELATI_DATA/podcast/hf --repo Scicom-intl/Melati > $C/dataset.log 2>&1
grep -h -i "warn" $C/*.log | grep -i -E "inter|font" | head -3 || true
ls $C/asr/figures $C/emotion/figures $C/nonverbal/figures $MELATI_DATA/podcast/hf/figures
echo CARDS_REDRAWN
