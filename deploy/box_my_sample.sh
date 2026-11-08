#!/bin/bash
# Redraw the Malaysian emotion labelling sample from an export, and restart the labelling page on it.
#
#   bash deploy/box_my_sample.sh $MELATI_DATA/pilot/export_rel
#
# Refuses once anyone has labelled: a redraw would orphan their answers. The page runs in its own tmux session
# (annotate, port 17900): the old page is stopped, because it loaded its sample at start and would keep serving it.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
EXPORT=${1:?usage: box_my_sample.sh <export dir>}
cd "$MELATI_REPO"
D=$MELATI_DATA/emo/my_test
if [ -s $D/labels.jsonl ]; then
    echo "REFUSED: $D/labels.jsonl has $(wc -l < $D/labels.jsonl) labels; move it aside first if a redraw is really wanted"
    exit 1
fi
$MELATI_ROOT/.venv-melati/bin/python emotion/sample_my_test.py --export $EXPORT --out $D/sample.jsonl \
    --summary $MELATI_DATA/results/emotion/my_test_sample.json
pkill -f 'emotion/annotate_server.py' || true          # the page alone; its session closes with it
for i in $(seq 1 20); do curl -sf http://127.0.0.1:17900/progress > /dev/null || break; sleep 1; done
S=annotate; tmux has-session -t $S 2>/dev/null && S=annotate-$(date +%m%d%H%M)
tmux new-session -d -s $S "cd $MELATI_REPO && $MELATI_ROOT/.venv-melati/bin/python emotion/annotate_server.py \
    --sample $D/sample.jsonl --labels $D/labels.jsonl --port 17900 2>&1 | tee $MELATI_LOGS/annotate.log"
for i in $(seq 1 30); do
    curl -sf http://127.0.0.1:17900/progress && echo && echo MYSAMPLE_DONE && exit 0
    sleep 1
done
echo "the labelling page did not come up; see $MELATI_LOGS/annotate.log"
exit 1
