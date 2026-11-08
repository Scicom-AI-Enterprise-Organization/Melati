#!/bin/bash
# Roll services of a running cluster onto the code now in the repo, one replica at a time, while a run uses them
# (serve/launch.py restart): the pipeline's client retries a dropped call on the other replicas.
#
#   bash deploy/box_roll.sh tag
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$MELATI_REPO
$MELATI_ROOT/.venv-melati/bin/python serve/launch.py restart --config "${CLUSTER:-serve/cluster.json}" --only "$@"
echo ROLL_DONE
