#!/bin/bash
# AudioSet's frame-level taggers (PretrainedSED) scored class by class where Melati has labels: VocalSound's test
# speakers, Expresso's read styles, PodcastFillers' test episodes and AMI's test meetings (nonverbal/audioset_eval.py).
#
#   bash deploy/box_audioset_eval.sh 1 BEATs
#   SETS="expresso podcasts" bash ...       (a rerun adds those sets to the results already there)
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-1}
cd "$MELATI_REPO"
export CUDA_VISIBLE_DEVICES=$G HF_HOME=$MELATI_DATA/hf
R=$MELATI_DATA/results/nonverbal
shift
for m in "${@:-BEATs}"; do
    $MELATI_ROOT/.venv-sed/bin/python nonverbal/audioset_eval.py --model $m --out $R/audioset_$m.json ${SETS:+--sets $SETS} \
        > $MELATI_LOGS/aseval_$m.log 2>&1
    grep -v -i warn $MELATI_LOGS/aseval_$m.log | tail -6
done
echo ASEVAL_DONE
