#!/bin/bash
# AudioSet's audio from agkphysics/AudioSet (Hugging Face, ungated, labels CC BY 4.0; the clips are YouTube's): parquet
# shards of the evaluation set, the balanced and the full unbalanced training set. SPLITS picks which; a re-run
# resumes, fetching only what is missing (scripts/hf_fetch.py: many files at once, each retried).
#
#   SPLITS='eval bal_train' bash deploy/box_audioset_fetch.sh
#   SPLITS=unbal_train bash deploy/box_audioset_fetch.sh   # 2.2 TB
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
D=$MELATI_DATA/audioset/agk
mkdir -p $D
cd "$MELATI_REPO"
for s in ${SPLITS:-eval bal_train}; do
    $MELATI_ROOT/.venv-melati/bin/python scripts/hf_fetch.py --repo agkphysics/AudioSet --include "data/$s/*" \
        --out $D --workers ${WORKERS:-24}
    echo "$s: $(ls $D/data/$s | wc -l) files, $(du -sh $D/data/$s | cut -f1)"
done
echo ASFETCH_DONE
