#!/bin/bash
# Re-write a released AudioSet tagger's card (audioset/release.py --card-only): every label with the model's average
# precision on it (audioset/per_class.py), grouped by AudioSet's ontology (fetched once, 343 KB). No GPU: the
# probabilities come from audioset/evaluate.py's cache and the card's Use snippet runs on the CPU. Without PUSH=1 it
# only writes the card into the release folder, to read before it goes out; with it, the card waits on a staging
# branch until its snippet runs from there.
#
#   bash deploy/box_audioset_card.sh $MELATI_DATA/models/audioset/frames1/clip.pt student_v3_fp32 \
#       Scicom-intl/Melati-AudioSet-EAT-Large
#   PUSH=1 MESSAGE="..." bash ...
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
CKPT=$1
RES=$MELATI_DATA/results/audioset/$2.json
REPO=$3
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=
PY=$MELATI_ROOT/.venv-sed/bin/python
ONT=$MELATI_DATA/audioset/ontology.json
[ -s $ONT ] || curl -fsSL https://raw.githubusercontent.com/audioset/ontology/master/ontology.json -o $ONT
$PY audioset/per_class.py --ckpt $CKPT --results $RES
$PY audioset/release.py --ckpt $CKPT --results $RES --out $MELATI_DATA/release/${REPO#*/} --repo $REPO \
    --card-only ${PUSH:+--push} --message "${MESSAGE:-Model card}"
echo ASCARD_DONE
