#!/bin/bash
# The sounds stage on a few hours of the full run before it joins the run (scripts/sounds_pilot.py): Melati's AudioSet
# tagger as a one-service cluster of its own (serve/cluster_sounds.json, which leaves the main cluster alone), the
# stage over recordings drawn from those the run has finished, then what their rows' `sounds` column says.
#
#   bash deploy/box_sounds_pilot.sh
#   RERUN=1 bash deploy/box_sounds_pilot.sh   # after a change
#
# Another tagger beside the one the run uses, on the same recordings, without touching the run's files: CLUSTER
# names its cluster and SHADOW=1 writes the stage into a work dir of its own ($P/work: links to each recording's files
# but sounds.jsonl), which the report then reads.
#
#   NAME=pilot50_ens SOURCES=$MELATI_DATA/podcast/sounds_pilot50/sources.jsonl \
#       CLUSTER=serve/cluster_sounds_ensemble.json SHADOW=1 bash deploy/box_sounds_pilot.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf
M=$MELATI_ROOT/.venv-melati/bin/python
NAME=${NAME:-pilot}         # a larger pilot: NAME=pilot50 HOURS=50; every finished recording: NAME=all HOURS=100000
CLUSTER=${CLUSTER:-serve/cluster_sounds.json}
P=$MELATI_DATA/podcast/sounds_$NAME
RUN_WORK=$MELATI_DATA/podcast/work
WORK=$RUN_WORK
mkdir -p $P
EP=$($M -c "import sys; sys.path.insert(0, 'serve'); import launch; print(launch.load(sys.argv[1])['endpoints'])" $CLUSTER)
$M serve/launch.py up --config $CLUSTER
[ -z "${SOURCES:-}" ] || [ -s $P/sources.jsonl ] || cp "$SOURCES" $P/sources.jsonl
[ -s $P/sources.jsonl ] || $M scripts/sounds_pilot.py pick --hours ${HOURS:-4} --out $P/sources.jsonl
if [ "${SHADOW:-0}" = 1 ]; then
    # links, never copies, and none for what the stage or the runner writes (sounds.jsonl, error.txt), so nothing
    # is written through a link into the run's own files
    WORK=$P/work
    $M - "$P/sources.jsonl" $RUN_WORK $WORK <<'EOF'
import json, sys
from pathlib import Path
src, run, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
for line in open(src):
    rid = json.loads(line)["id"]
    (work / rid).mkdir(parents=True, exist_ok=True)
    for f in (run / rid).iterdir():
        link = work / rid / f.name
        if f.name not in ("sounds.jsonl", "error.txt") and not link.is_symlink():
            link.symlink_to(f)
EOF
fi
if [ -n "${RERUN:-}" ]; then
    # the server onto new code, and the stage again over the same recordings: a stage whose file exists is skipped
    $M serve/launch.py restart --config $CLUSTER
    $M - "$P/sources.jsonl" $WORK <<'EOF'
import json, sys
from pathlib import Path
for line in open(sys.argv[1]):
    (Path(sys.argv[2]) / json.loads(line)["id"] / "sounds.jsonl").unlink(missing_ok=True)
EOF
fi
$M -m melati.run --sources $P/sources.jsonl --work $WORK --stages sounds \
    --endpoints $EP --recordings ${RECS:-16} --timing $P/timing.jsonl
[ "${REPORT:-1}" = 0 ] || $M scripts/sounds_pilot.py report --sources $P/sources.jsonl --work $WORK \
    --timing $P/timing.jsonl --out $MELATI_DATA/results/sounds/$NAME.json
echo SOUNDS_PILOT_DONE
