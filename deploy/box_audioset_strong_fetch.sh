#!/bin/bash
# AudioSet's strongly labelled subset as CLAPv2/audioset_strong ships it (Hugging Face, ungated; labels from Google's
# strong annotations, CC BY 4.0; the clips are YouTube's): about 102k training clips with every event timed, 512
# validation and 16.9k test clips (the test clips come from AudioSet's evaluation set: they score, never train). Fetched
# many files at a time, then every file checked against the repo's sha256 (scripts/hf_fetch.py --verify).
#
#   bash deploy/box_audioset_strong_fetch.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
D=$MELATI_DATA/audioset/strong
mkdir -p $D
cd "$MELATI_REPO"
P=$MELATI_ROOT/.venv-melati/bin/python
$P scripts/hf_fetch.py --repo CLAPv2/audioset_strong --include "data/*" --out $D --workers ${WORKERS:-32}
$P scripts/hf_fetch.py --repo CLAPv2/audioset_strong --include "data/*" --out $D --workers 8 --verify
du -sh $D/data/*
echo ASSTRONG_DONE
