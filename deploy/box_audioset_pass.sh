#!/bin/bash
# AudioSet's taggers (the average of PretrainedSED's BEATs, ATST-F and M2D) over every finished Melati recording,
# one slice a GPU, into each recording's audioset.jsonl (nonverbal/audioset_pass.py). A recording already done is
# skipped, so this can run again as the full run finishes more (deploy/box_full_publish.sh runs it before every
# publish). Uses the memory the model servers leave free. Fails if a slice does.
#
#   bash deploy/box_audioset_pass.sh 0 1 2 4 5
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
[ $# -gt 0 ] || set -- 0 1 2 4 5
GPUS=("$@")
n=${#GPUS[@]}
L=${ASPASS_LOGS:-$MELATI_LOGS}
pids=()
for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $MELATI_ROOT/.venv-sed/bin/python nonverbal/audioset_pass.py \
        --work $MELATI_DATA/podcast/work --slice $i/$n > $L/aspass_$i.log 2>&1 &
    pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
grep -h "AUDIOSET_PASS_DONE\|Error" $L/aspass_*.log | sort | uniq -c
echo "recordings tagged: $(find $MELATI_DATA/podcast/work -maxdepth 2 -name audioset.jsonl | wc -l)"
[ "$failed" = 0 ] || { echo "ASPASS_FAILED: $failed of $n slices"; exit 1; }
echo ASPASS_DONE
