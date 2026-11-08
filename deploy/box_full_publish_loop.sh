#!/bin/bash
# Publish Melati v1 as the full run finishes it: deploy/box_full_publish.sh every EVERY_H hours (default 6) until
# deploy/box_full.sh has finished (it leaves logs/FULL_GPU_DONE; a driver stopped and restarted mid-run does not),
# then once more with FINAL=1, which packs every block and leaves out what failed for good. Each publish's log is
# logs/publish_<time>.log.
#
#   bash deploy/box_full_publish_loop.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
EVERY_H=${EVERY_H:-6}
L=$MELATI_DATA/podcast/logs
running() { [ ! -e "$L/FULL_GPU_DONE" ]; }
publish() {
  local log=$L/publish_$(date +%m%d%H%M).log
  FINAL=${1:-0} bash $MELATI_REPO/deploy/box_full_publish.sh > "$log" 2>&1
  echo "$(date +%H:%M) publish (FINAL=${1:-0}) exit $?: $(grep -E '^\[hf\] \{|FULL_PUBLISH_DONE|CHECK_FAILED|Error' "$log" | tail -2 | tr '\n' ' ')"
}
while running; do
  publish 0
  for _ in $(seq 1 $((EVERY_H * 60))); do running || break; sleep 60; done
done
publish 1
echo PUBLISH_LOOP_DONE
