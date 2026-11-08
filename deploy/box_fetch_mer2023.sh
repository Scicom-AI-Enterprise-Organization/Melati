#!/bin/bash
# Fetch MER2023 (gated; licence accepted on Hugging Face 2026-10-23), verify every part, and extract only the
# labelled clips as 16 kHz mono audio.
#
#   bash deploy/box_fetch_mer2023.sh
#
# ~140 GB of archives (MER-SEMI's 73k unlabelled videos travel with the training set). Resumable: a verified part is
# never fetched again, and extraction skips clips already converted.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1
D=$MELATI_DATA/emo/raw/mer2023
M=$MELATI_ROOT/.venv-melati/bin/python
$M emotion/fetch_mer2023.py --root $D download
$M emotion/fetch_mer2023.py --root $D list
$M emotion/fetch_mer2023.py --root $D extract
echo MER2023_DONE
