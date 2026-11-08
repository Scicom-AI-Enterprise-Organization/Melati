#!/bin/bash
# The licence-clean non-verbal models into the pipeline, and everything the paper and the card read re-run with them.
#
#   gap head      HEAD (default rel4: PodcastFillers' CC BY episodes, AMI's candidates, COUGHVID and Coswara), tagging
#                 BREATHS only, at the threshold chosen on PodcastFillers' validation episodes (gap_eval_<HEAD>pf.json)
#   frame model   v3 (AMI and PodcastFillers' CC BY episodes), tagging LAUGHS and COUGHS: laugh threshold from
#                 PodcastFillers' validation episodes, cough threshold from AMI's dev meetings (frames3_eval_pf.json).
#                 It finds more laughs than any gap head on every test set, and the coughs no gap head finds; the gap
#                 heads' extra laugh tags in podcasts sound to AudioSet like gasps.
#
# The head they replace (rel2) also trained on NonverbalTTS's candidates, whose labels are CC BY-NC-SA 4.0: that
# cannot share one set of weights with PodcastFillers' licence. Then: PodcastFillers' test episodes through the pipeline
# with both models, the AudioSet audit, and the 200-podcast pilot (tag counts and the comparison with Emilia-Pipe).
#
#   bash deploy/box_nv_ship_rel4.sh 4 rel4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
HEAD=${2:-rel4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
MD=$MELATI_DATA/models
R=$MELATI_DATA/results
B=$MELATI_DATA/bench/pf_test
P=$MELATI_DATA/pilot
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'
FT=$($M -c "import json; d = json.load(open('$R/nonverbal/frames3_eval_pf.json')); t = {'laugh': (d.get('podcastfillers') or {}).get('threshold'), 'cough': d['ami']['cough']['threshold']}; print(json.dumps({k: v for k, v in t.items() if v}))")
echo "gap head $HEAD (breaths), frame thresholds $FT"

# 1. the pipeline's model directories
[ -f $MD/nv_frames1/whisper.pt ] || { mkdir -p $MD/nv_frames1 && cp $MD/nv_frames/whisper.pt $MD/nv_frames1/; }
cp $MD/nv_frames3/whisper.pt $MD/nv_frames/whisper.pt
cp $MD/nv_heads_$HEAD/whisper.pt $MD/nv_heads/whisper.pt

# 2. PodcastFillers' test episodes, gap head and frame model together
$M nonverbal/relabel.py --work $B/work_rel --sources $B/sources.jsonl --models whisper --variant "$RULE" \
    --thresholds $R/nonverbal/gap_eval_${HEAD}pf.json --classes breath --rule "$RULE" \
    --new-work $B/work_nvrel4f --score-pf $R/fillers/podcastfillers_nvrel4f.json --frames whisper \
    --frame-thresholds "$FT" > $LOG/nvship4_pf.log 2>&1
grep -E "thresholds|frames:|candidates|RELABEL_DONE" $LOG/nvship4_pf.log
$M scripts/audit_nv_tags.py --work $B/work_nvrel4f --out $R/fillers/audit_nv_tags_rel4f.json 2>&1 | grep -E "^breath|AUDIT_DONE"

# 3. the pilot: tags per hour, and the comparison with Emilia-Pipe
$M nonverbal/relabel.py --work $P/work_rel --sources $P/sources.jsonl --models whisper --variant "$RULE" \
    --thresholds $R/nonverbal/gap_eval_${HEAD}pf.json --classes breath --rule "$RULE" \
    --new-work $P/work_nvrel4 --export-out /dev/shm/melati_nv/pilot_export_nvrel4 --assemble --frames whisper \
    --frame-thresholds "$FT" > $LOG/nvship4_pilot.log 2>&1
grep -E "frames:|candidates|RELABEL_DONE" $LOG/nvship4_pilot.log
$M scripts/count_events.py $P/work_nvrel4
$M serve/launch.py up --config serve/cluster_compare.json
$M scripts/compare_emilia.py --sources $P/sources.jsonl --melati /dev/shm/melati_nv/pilot_export_nvrel4 \
    --emilia $P/emilia_in_processed --work $P/work_nvrel4 --endpoints $MELATI_DATA/serve_compare/endpoints.json \
    --out $R/compare/pilot_nvrel4.json > $LOG/nvship4_compare.log 2>&1
$M serve/launch.py down --config serve/cluster_compare.json
rm -rf /dev/shm/melati_nv
echo NV_SHIP4_DONE
