#!/bin/bash
# Restart the full run's clients onto the code now in the repo: every melati.run of deploy/box_full.sh, and its worker
# processes, is stopped, and the driver's next pass starts fresh ones over what is left. Stage outputs already written
# stay; each recording in flight redoes the stage it was in. The model servers are not touched.
#
#   bash deploy/box_full_clients_restart.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
for pid in $(pgrep -f "melati[.]run --sources $MELATI_DATA/podcast/sources_"); do
  pkill -TERM -P "$pid" 2>/dev/null
  kill -TERM "$pid" 2>/dev/null && echo "stopped client $pid"
done
