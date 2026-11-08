#!/bin/bash
# Distil the Melati AudioSet tagger (audioset/train.py) on a pool (audioset/pool.py), one process a GPU, on the memory the
# model servers leave free. POOL names the pool, SOURCES the cache folders it is built from (once), RUN the output.
#
#   POOL=bal SOURCES=bal_train RUN=pilot_bal EPOCHS=3 bash deploy/box_audioset_train.sh
#   POOL=big SOURCES='strong_train bal_train unbal_train' POOL_ARGS=--index-only RUN=big_from_v1 EPOCHS=3 \
#       ARGS="--init melati:$MELATI_DATA/models/audioset/clean_e10/best.pt --eval-every 2000" bash ...
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf OMP_NUM_THREADS=4
V=$MELATI_ROOT/.venv-sed/bin
C=$MELATI_DATA/audioset/cache
[ -s $C/pool_${POOL}_meta.json ] || $V/python audioset/pool.py --sources $SOURCES --name $POOL ${POOL_ARGS:-}
cat $C/pool_${POOL}_meta.json
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,4,5} $V/torchrun --nproc_per_node $(echo ${GPUS:-0,1,2,4,5} | tr , ' ' | wc -w) \
    --master_port ${PORT:-29511} audioset/train.py --pool $POOL --out $MELATI_DATA/models/audioset/$RUN \
    --epochs ${EPOCHS:-10} ${ARGS:-}
echo ASTRAIN_DONE
