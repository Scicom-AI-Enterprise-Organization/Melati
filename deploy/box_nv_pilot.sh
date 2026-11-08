#!/bin/bash
# The 200-podcast pilot with gap candidates labelled by the non-verbal head instead of Qwen3-Omni (nonverbal/
# relabel.py: everything else is the release model's run, linked), re-exported and compared with Emilia-Pipe again
# by scripts/compare_emilia.py, so the paper's pilot numbers are the pipeline as it now ships.
#
#   bash deploy/box_nv_pilot.sh 4
#
# Needs the eval cluster (align, front) up: serve/cluster_eval.json. The export goes to /dev/shm (~8k small files).
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
P=$MELATI_DATA/pilot
X=/dev/shm/melati_nv/pilot_export_nvwhisper
LOG=$MELATI_LOGS
$M nonverbal/relabel.py --work $P/work_rel --sources $P/sources.jsonl --models whisper --variant whisper \
    --thresholds $MELATI_DATA/results/nonverbal/nonverbaltts_gaps_heads.json --new-work $P/work_nvwhisper \
    --export-out $X --assemble > $LOG/nvpilot_relabel.log 2>&1
grep -E "candidates|re-exported|RELABEL_DONE" $LOG/nvpilot_relabel.log
$M scripts/compare_emilia.py --sources $P/sources.jsonl --melati $X --emilia $P/emilia_in_processed \
    --work $P/work_nvwhisper --endpoints $MELATI_DATA/serve_eval/endpoints.json \
    --out $MELATI_DATA/results/compare/pilot_nvwhisper.json > $LOG/nvpilot_compare.log 2>&1
tail -3 $LOG/nvpilot_compare.log
echo NV_PILOT_DONE
