#!/bin/bash
# The looser gap finder (FLOOR_DB 2, melati/nv_gaps.py) and the head retrained on what it finds (nv_heads_v3) made
# the default, then everything that reads candidates re-scored: the miss diagnosis under both rules, PodcastFillers'
# test episodes through export (candidates found again from each segment's vocal stem), the AudioSet audit of the
# breath tags it cannot score, and the 200-podcast pilot against Emilia-Pipe.
#
#   bash deploy/box_nv_rule_apply.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
R=$MELATI_DATA/results/nonverbal
F=$MELATI_DATA/results/fillers
MD=$MELATI_DATA/models
P=$MELATI_DATA/pilot
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'
$M serve/launch.py down --config serve/cluster_align.json > /dev/null 2>&1 || true

$M nonverbal/gap_rule.py diagnose --which test --rules '{"floor_db": 6}' --out $R/gap_diagnose_test_old.json \
    2>&1 | grep -vE "warn" | head -4
$M nonverbal/gap_rule.py diagnose --which test --rules "$RULE" --out $R/gap_diagnose_test_new.json \
    2>&1 | grep -vE "warn" | head -4

[ -f $MD/nv_heads_v2/whisper.pt ] || { mkdir -p $MD/nv_heads_v2 && cp $MD/nv_heads/whisper.pt $MD/nv_heads_v2/; }
cp $MD/nv_heads_v3/whisper.pt $MD/nv_heads/whisper.pt

$M nonverbal/relabel.py --work $MELATI_DATA/bench/pf_test/work_rel --sources $MELATI_DATA/bench/pf_test/sources.jsonl \
    --models whisper --variant "$RULE" --thresholds $R/gap_eval_v3head.json --classes breath laugh cough \
    --rule "$RULE" --new-work $MELATI_DATA/bench/pf_test/work_nvwhisper3 \
    --score-pf $F/podcastfillers_nvwhisper3.json > $LOG/nvapply_pf.log 2>&1
grep -E "thresholds|candidates|RELABEL_DONE" $LOG/nvapply_pf.log
$M scripts/audit_nv_tags.py --work $MELATI_DATA/bench/pf_test/work_nvwhisper3 \
    --out $F/audit_nv_tags_v3.json 2>&1 | grep -E "^breath|AUDIT_DONE"

$M nonverbal/relabel.py --work $P/work_rel --sources $P/sources.jsonl --models whisper --variant "$RULE" \
    --thresholds $R/gap_eval_v3head.json --classes breath laugh cough --rule "$RULE" \
    --new-work $P/work_nvwhisper3 --export-out /dev/shm/melati_nv/pilot_export_nvwhisper3 --assemble \
    > $LOG/nvapply_pilot.log 2>&1
grep -E "candidates|RELABEL_DONE" $LOG/nvapply_pilot.log
$M serve/launch.py up --config serve/cluster_compare.json
$M scripts/compare_emilia.py --sources $P/sources.jsonl --melati /dev/shm/melati_nv/pilot_export_nvwhisper3 \
    --emilia $P/emilia_in_processed --work $P/work_nvwhisper3 --endpoints $MELATI_DATA/serve_compare/endpoints.json \
    --out $MELATI_DATA/results/compare/pilot_nvwhisper3.json > $LOG/nvapply_compare.log 2>&1
$M serve/launch.py down --config serve/cluster_compare.json
rm -rf /dev/shm/melati_nv
echo NV_APPLY_DONE
