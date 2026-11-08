#!/bin/bash
# The sounds labelling set: the dataset's own non-verbal tags, fillers and untagged gaps, sampled from the full run's
# finished recordings (nonverbal/sample_my_sounds.py), then the labelling page restarted with both tasks (emotion at /,
# sounds at /sounds; port 17900, tmux session `annotate`). Answers already given stay in their labels files.
#
#   bash deploy/box_my_sounds.sh
#   then the page is on the box's port 17900: http://<box>:17900/sounds/
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export PYTHONPATH=$MELATI_REPO HF_HOME=$MELATI_DATA/hf
M=$MELATI_ROOT/.venv-melati/bin/python
D=$MELATI_DATA/nv/my_sounds
E=$MELATI_DATA/emo/my_test
[ -s "$D/sample.jsonl" ] || $M nonverbal/sample_my_sounds.py --sources $MELATI_DATA/podcast/sources.jsonl \
    --work $MELATI_DATA/podcast/work --out "$D" --summary $MELATI_DATA/results/nonverbal/my_sounds_sample.json
pkill -f 'emotion/annotate_server[.]py' || true
sleep 2
S=annotate; tmux has-session -t $S 2>/dev/null && S=annotate-$(date +%m%d%H%M)
tmux new-session -d -s $S "cd $MELATI_REPO && $M emotion/annotate_server.py --sample $E/sample.jsonl \
    --labels $E/labels.jsonl --sounds-sample $D/sample.jsonl --sounds-labels $D/labels.jsonl --port 17900 2>&1 \
    | tee $MELATI_LOGS/annotate.log"
for _ in $(seq 1 30); do
  curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:17900/sounds/ 2>/dev/null | grep -q 200 && { echo "LABELLING_UP (tmux $S)"; exit 0; }
  sleep 1
done
echo "the labelling page did not come up; see $MELATI_LOGS/annotate.log"; exit 1
