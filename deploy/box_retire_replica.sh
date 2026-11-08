#!/bin/bash
# Stop one replica of a running cluster for good, after its entry in the cluster config was set to null (a retired
# replica): its process group is stopped and the endpoints file rewritten, which the pipeline's client re-reads.
#
#   bash deploy/box_retire_replica.sh align 0
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SVC=${1:?service}; K=${2:?replica number}; CLUSTER=${CLUSTER:-serve/cluster.json}
cd "$MELATI_REPO"
RUN=$($MELATI_ROOT/.venv-melati/bin/python -c "import sys; sys.path.insert(0, 'serve'); import launch; print(launch.load(sys.argv[1])['run_dir'])" "$CLUSTER")
PIDF=$RUN/$SVC-$K.pid
[ -s "$PIDF" ] && kill -TERM -"$(cat "$PIDF")" 2>/dev/null && echo "stopped $SVC#$K (pid $(cat "$PIDF"))"
rm -f "$PIDF"
sleep 5
$MELATI_ROOT/.venv-melati/bin/python -c "import sys; sys.path.insert(0, 'serve'); import launch; launch.write_endpoints(launch.load(sys.argv[1]))" "$CLUSTER"
