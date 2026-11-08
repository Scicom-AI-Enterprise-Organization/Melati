#!/bin/bash
# Melati's non-verbal heads end to end: training clips, features on three encoders, a head on each, then the two
# evaluations Qwen3-Omni was scored by (NonverbalTTS's gap candidates; PodcastFillers' test episodes re-exported).
#
#   bash deploy/box_nv_heads.sh 4 3 5
#
# Args: one GPU per encoder's feature extraction (whisper, wavlm, ast); training and evaluation use the first.
# Each step skips work already done (the manifest, a corpus's features) and logs to its own file.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G1=${1:-4}; G2=${2:-3}; G3=${3:-5}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
LOG=$MELATI_LOGS
mkdir -p $R $MELATI_DATA/models/nv_heads

# the clips live in /dev/shm (a network disk creates ~7 small files a second): rebuilt if the box lost them
[ -s $NV/manifest.jsonl ] && [ -d /dev/shm/melati_nv/clips ] || $M nonverbal/build_manifest.py --podcastfillers $MELATI_DATA/bench/podcastfillers \
    --vocalsound $NV/raw/vocalsound/vs_release_44k.zip --out $NV > $LOG/nv_manifest.log 2>&1
tail -1 $LOG/nv_manifest.log

pids=""
for pair in whisper:$G1 wavlm:$G2 ast:$G3; do
    enc=${pair%%:*}; gpu=${pair##*:}
    CUDA_VISIBLE_DEVICES=$gpu $M emotion/extract.py --manifest $NV/manifest.jsonl --encoder $enc \
        --out $NV/feats > $LOG/nv_extract_$enc.log 2>&1 &
    pids="$pids $!"
done
for p in $pids; do wait $p; done
grep -h "EXTRACT_DONE" $LOG/nv_extract_*.log

for enc in whisper wavlm ast; do
    CUDA_VISIBLE_DEVICES=$G1 $M nonverbal/train.py --manifest $NV/manifest.jsonl --feats $NV/feats --encoder $enc \
        --out $R/head_$enc.json --save-model $MELATI_DATA/models/nv_heads/$enc.pt \
        --save-probs $R/head_${enc}_test.npz > $LOG/nv_train_$enc.log 2>&1
    grep -E "test all|TRAIN_DONE" $LOG/nv_train_$enc.log
done

CUDA_VISIBLE_DEVICES=$G1 $M nonverbal/eval_gaps.py --candidates $R/nonverbaltts_gaps.json --models whisper wavlm ast \
    --out $R/nonverbaltts_gaps_heads.json > $LOG/nv_eval_gaps.log 2>&1
grep -E "^==|^   " $LOG/nv_eval_gaps.log

CUDA_VISIBLE_DEVICES=$G1 $M nonverbal/relabel.py --work $MELATI_DATA/bench/pf_test/work_rel \
    --sources $MELATI_DATA/bench/pf_test/sources.jsonl --models whisper wavlm ast \
    --thresholds $R/nonverbaltts_gaps_heads.json --new-work $MELATI_DATA/bench/pf_test/work_nvheads \
    --out $MELATI_DATA/results/fillers/podcastfillers_nvheads.json > $LOG/nv_rescore_pf.log 2>&1
tail -2 $LOG/nv_rescore_pf.log
echo NV_HEADS_DONE
