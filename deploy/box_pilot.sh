#!/bin/bash
# Run Melati over a source list: bring the model servers up, then drive the pipeline against them.
#
#   bash deploy/box_pilot.sh <sources.jsonl> <work> <export> [cluster.json] [stages...]
#   bash deploy/box_pilot.sh $MELATI_DATA/pilot/sources.jsonl $MELATI_DATA/pilot/work \
#        $MELATI_DATA/pilot/export serve/cluster.json
#
# GPUs are only ever touched by the servers in the cluster config; the pipeline process is a client.
# A run's summary (wall clock, hours, realtime factor, seconds per stage) is appended to
# <work>/../timing.jsonl, which is where the throughput numbers come from.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SRC=${1:?sources.jsonl}; WORK=${2:?work dir}; OUT=${3:?export dir}; CLUSTER=${4:-serve/cluster.json}
shift 4 2>/dev/null || shift $#
STAGES=${*:-front diar seg asr align quality tag fill export}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
$M serve/launch.py up --config "$CLUSTER"
ENDPOINTS=$($M -c "import sys; sys.path.insert(0, 'serve'); import launch; print(launch.load(sys.argv[1])['endpoints'])" "$CLUSTER")
$M -m melati.run --sources "$SRC" --work "$WORK" --out "$OUT" --endpoints "$ENDPOINTS" \
    --stages $STAGES --timing "$(dirname "$WORK")/timing.jsonl"
echo PIPELINE_DONE
