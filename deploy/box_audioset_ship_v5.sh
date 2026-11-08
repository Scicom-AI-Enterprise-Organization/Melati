#!/bin/bash
# Melati-AudioSet-EAT-Large v5: v4 trained 3 epochs more toward six teachers, CED-base, EAT-large and SSLAM with BEATs
# and its own v3 and v4 (deploy/box_audioset_v5.sh), which beat v4 on the evaluation set. Every model in fp32 (from
# audioset/evaluate.py's cache) with the two teacher averages the card names, v5's AP per class
# (audioset/per_class.py), then audioset/release.py, which sets v4 and v3 beside it. Without PUSH=1 it only builds the
# folder and its card, to read before anything goes out; with it, v5 waits on a staging branch until it scores from the
# Hub.
#
#   bash deploy/box_audioset_ship_v5.sh
#   PUSH=1 bash deploy/box_audioset_ship_v5.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3}
PY=$MELATI_ROOT/.venv-sed/bin/python
M=$MELATI_DATA/models/audioset
R=$MELATI_DATA/results/audioset
V3=melati:$M/frames1/clip.pt
V4=melati:$M/frames1_fw0/clip.pt
V5=$M/v5_big6/best.pt
[ -s $R/student_v5_release_fp32.json ] || $PY audioset/evaluate.py --fp32 --models ced-base eat-large eat-base sslam ast \
    beats melati:$M/clean_e10/best.pt melati:$M/big_from_v1/best.pt $V3 $V4 melati:$V5 \
    --averages ced-base,eat-large,sslam beats,ced-base,eat-large,sslam,$V3,$V4 --out $R/student_v5_release_fp32.json
[ -s $R/per_class_v5_big6.json ] || $PY audioset/per_class.py --ckpt $V5 --results $R/student_v5_release_fp32.json
$PY audioset/release.py --ckpt $V5 --results $R/student_v5_release_fp32.json --repo Scicom-intl/Melati-AudioSet-EAT-Large \
    --out $MELATI_DATA/release/Melati-AudioSet-EAT-Large --name "$V4=previous release (v4)" \
    --name "$V3=an earlier release (v3)" ${PUSH:+--push} --message "v5: trained toward six teachers, BEATs and its own v3 \
and v4 among them, $($PY -c "import json;print(json.load(open('$R/student_v5_release_fp32.json'))['models']['melati:$V5'])") mAP"
echo SHIP_V5_DONE
