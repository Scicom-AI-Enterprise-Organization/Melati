#!/bin/bash
# AudioSet's evaluation set decoded once (audioset/data.py), then every candidate teacher scored on it, alone and in
# every average (audioset/evaluate.py). Runs in .venv-sed, on the memory the model servers leave free.
#
#   bash deploy/box_audioset_teachers.sh 1
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-1}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf CUDA_VISIBLE_DEVICES=$G
V=$MELATI_ROOT/.venv-sed/bin/python
[ -s $MELATI_DATA/audioset/cache/eval_meta.json ] || $V audioset/data.py cache --split eval --workers 32
cat $MELATI_DATA/audioset/cache/eval_meta.json
$V audioset/evaluate.py --models ${MODELS:-ced-base eat-large eat-base sslam ast} --split eval \
    --out $MELATI_DATA/results/audioset/teachers_eval.json
echo ASTEACH_DONE
