#!/bin/bash
# Melati-AudioSet-EAT-Large v4: stage 3 trained again with the frame loss off (deploy/box_audioset_control.sh), which
# beat v3 on the evaluation set. Every model in fp32 (audioset/evaluate.py, from its cache), v4's AP per class
# (audioset/per_class.py), then audioset/release.py, which sets v3 beside it. Without PUSH=1 it only builds the folder
# and its card, to read before anything goes out; with it, v4 waits on a staging branch until it scores from the Hub.
#
#   bash deploy/box_audioset_ship_v4.sh
#   PUSH=1 bash deploy/box_audioset_ship_v4.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3}
PY=$MELATI_ROOT/.venv-sed/bin/python
M=$MELATI_DATA/models/audioset
R=$MELATI_DATA/results/audioset
V3=melati:$M/frames1/clip.pt
V4=$M/frames1_fw0/clip.pt
[ -s $R/student_v4_fp32.json ] || $PY audioset/evaluate.py --fp32 --models ced-base eat-large eat-base sslam ast \
    melati:$M/clean_e10/best.pt melati:$M/big_from_v1/best.pt $V3 melati:$V4 --out $R/student_v4_fp32.json
[ -s $R/per_class_frames1_fw0.json ] || $PY audioset/per_class.py --ckpt $V4 --results $R/student_v4_fp32.json
$PY audioset/release.py --ckpt $V4 --results $R/student_v4_fp32.json --repo Scicom-intl/Melati-AudioSet-EAT-Large \
    --out $MELATI_DATA/release/Melati-AudioSet-EAT-Large --name "$V3=previous release (v3)" --with-frames $V3 \
    ${PUSH:+--push} --message "v4: stage 3 again without the frame loss, $(
        $PY -c "import json;print(json.load(open('$R/student_v4_fp32.json'))['models']['melati:$V4'])") mAP"
echo SHIP_V4_DONE
