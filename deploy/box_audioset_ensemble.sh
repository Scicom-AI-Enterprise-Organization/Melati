#!/bin/bash
# Melati-AudioSet-Ensemble: six AudioSet taggers averaged (CED-base, EAT-large, SSLAM, BEATs, and
# Melati-AudioSet-EAT-Large's v3 and v4; 0.5218 mAP in results/audioset/teachers_beats_fp32.json), built, uploaded
# private, checked from the Hub and published by audioset/release_ensemble.py (PUSH=1; without it, only built). BEATs's
# code goes out with its licence (microsoft/unilm, MIT). The check holds all six members on one GPU (EVAL_GPU), about
# 5 GB in fp32.
#
#   PUSH=1 bash deploy/box_audioset_ensemble.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf OMP_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
E=$MELATI_ROOT/ext/beats
[ -s $E/LICENSE ] || curl -fsSL --retry 3 https://raw.githubusercontent.com/microsoft/unilm/master/LICENSE -o $E/LICENSE
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3} $MELATI_ROOT/.venv-sed/bin/python audioset/release_ensemble.py \
    --out $MELATI_DATA/release/Melati-AudioSet-Ensemble --repo Scicom-intl/Melati-AudioSet-Ensemble ${PUSH:+--push} \
    ${CARD_ONLY:+--card-only}
echo ASENS_DONE
