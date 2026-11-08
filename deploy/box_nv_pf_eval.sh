#!/bin/bash
# One non-verbal head through the whole pipeline on PodcastFillers' test episodes (nonverbal/relabel.py: the same run,
# its candidates labelled by this head at the thresholds chosen for it on NonverbalTTS), scored against
# PodcastFillers' hand labels, and the AudioSet audit of the breath tags PodcastFillers has no label for.
#
#   bash deploy/box_nv_pf_eval.sh rel5 3
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
V=$1
G=${2:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
export MELATI_NV_HEADS=$MELATI_DATA/models/nv_heads_$V
M=$MELATI_ROOT/.venv-melati/bin/python
R=$MELATI_DATA/results
B=$MELATI_DATA/bench/pf_test
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'
$M nonverbal/relabel.py --work $B/work_rel --sources $B/sources.jsonl --models whisper --variant "$RULE" \
    --thresholds $R/nonverbal/gap_eval_$V.json --classes breath laugh cough --rule "$RULE" \
    --new-work $B/work_nv$V --score-pf $R/fillers/podcastfillers_nv$V.json > $LOG/nvpf_$V.log 2>&1
grep -E "thresholds|candidates|RELABEL_DONE" $LOG/nvpf_$V.log
$M scripts/audit_nv_tags.py --work $B/work_nv$V --out $R/fillers/audit_nv_tags_$V.json 2>&1 | grep -E "^breath|AUDIT_DONE"
echo NV_PF_DONE
