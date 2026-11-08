#!/bin/bash
# The ASR-independent half of a Melati run (front, diar, seg), on GPU 5 beside the eval cluster,
# while GPUs 0-4 train the ASR sweep. The later stages (asr onwards) run once the checkpoint is
# chosen; nothing here depends on it, and every stage resumes from its own files.
#
#   bash deploy/box_pilot_front.sh <dir>
#
# <dir> holds sources.jsonl (scripts/make_pilot.py). Uses the eval cluster's `front` server
# (serve/cluster_eval.json, already up) and starts a diarization server (serve/cluster_pilot5.json).
# Timing on a shared GPU is not a throughput number; that is measured separately.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
P=${1:?dir with sources.jsonl}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
$M serve/launch.py up --config serve/cluster_pilot5.json
$M -m melati.run --sources "$P/sources.jsonl" --work "$P/work" --out "$P/export" \
    --endpoints $MELATI_DATA/serve_eval/endpoints.json $MELATI_DATA/serve_pilot5/endpoints.json \
    --stages front diar seg --timing "$P/timing_front.jsonl"
echo PILOT_FRONT_DONE
