#!/bin/bash
# The pipeline switched to the head that is published (nv_heads_rel2 -> nv_heads; the VocalSound head stays in
# nv_heads_v3/), then what the paper reads re-run with it: the AudioSet audit of PodcastFillers' unscored breath tags,
# and the 200-podcast pilot (tag counts and the comparison with Emilia-Pipe).
#
#   bash deploy/box_nv_ship_release.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
MD=$MELATI_DATA/models
R=$MELATI_DATA/results
P=$MELATI_DATA/pilot
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'
cp $MD/nv_heads_rel2/whisper.pt $MD/nv_heads/whisper.pt
$M scripts/audit_nv_tags.py --work $MELATI_DATA/bench/pf_test/work_nvrel2 --out $R/fillers/audit_nv_tags_rel2.json \
    2>&1 | grep -E "^breath|AUDIT_DONE"
$M nonverbal/relabel.py --work $P/work_rel --sources $P/sources.jsonl --models whisper --variant "$RULE" \
    --thresholds $R/nonverbal/gap_eval_rel2.json --classes breath laugh cough --rule "$RULE" \
    --new-work $P/work_nvrel2 --export-out /dev/shm/melati_nv/pilot_export_nvrel2 --assemble > $LOG/nvship_pilot.log 2>&1
grep -E "candidates|RELABEL_DONE" $LOG/nvship_pilot.log
$M scripts/count_events.py $P/work_nvrel2
$M serve/launch.py up --config serve/cluster_compare.json
$M scripts/compare_emilia.py --sources $P/sources.jsonl --melati /dev/shm/melati_nv/pilot_export_nvrel2 \
    --emilia $P/emilia_in_processed --work $P/work_nvrel2 --endpoints $MELATI_DATA/serve_compare/endpoints.json \
    --out $R/compare/pilot_nvrel2.json > $LOG/nvship_compare.log 2>&1
$M serve/launch.py down --config serve/cluster_compare.json
rm -rf /dev/shm/melati_nv
echo NV_SHIP_DONE
