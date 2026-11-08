#!/bin/bash
# Stop deploy/box_full.sh (the driver, its watchdog) and its clients, leaving every model server running, so the run
# can be started again with new settings or new client code and fresh retry passes:
#
#   bash deploy/box_full_stop_driver.sh
#   REC=32 bash deploy/box_full.sh
#
# Every stage output already written stays; each recording in flight redoes the stage it was in.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
pkill -TERM -f 'deploy/box_full[.]sh' && echo "stopped the driver and its watchdog"
sleep 2
bash $MELATI_REPO/deploy/box_full_clients_restart.sh
sleep 5
left=$(pgrep -f "melati[.]run --sources $MELATI_DATA/podcast/sources_" | wc -l)
echo "client processes left: $left"
