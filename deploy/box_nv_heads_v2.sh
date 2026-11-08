#!/bin/bash
# The non-verbal heads again, now also trained on real gap candidates (nonverbal/candidates.py): PodcastFillers'
# training episodes, and NonverbalTTS's training clips outside the 800 that choose the thresholds. The heads trained
# on hand-cut events alone (deploy/box_nv_heads.sh) are kept beside them, as nv_heads_events/ and *_events.json.
#
#   bash deploy/box_nv_heads_v2.sh 4 3 5
#
# Args: one GPU per encoder's feature extraction (whisper, wavlm, ast); training and evaluation use the first.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G1=${1:-4}; G2=${2:-3}; G3=${3:-5}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
F=$MELATI_DATA/results/fillers
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS

if [ ! -d $MD/nv_heads_events ]; then                     # the events-only heads, and what they scored
    cp -r $MD/nv_heads $MD/nv_heads_events
    for e in whisper wavlm ast; do cp $R/head_$e.json $R/head_${e}_events.json; done
    cp $R/nonverbaltts_gaps_heads.json $R/nonverbaltts_gaps_heads_events.json
    cp $F/podcastfillers_nvheads.json $F/podcastfillers_nvheads_events.json
fi
cat $NV/manifest.jsonl $NV/manifest_pfc.jsonl $NV/manifest_nvc.jsonl > $NV/manifest_v2.jsonl
wc -l $NV/manifest_v2.jsonl

pids=""
for pair in whisper:$G1 wavlm:$G2 ast:$G3; do
    enc=${pair%%:*}; gpu=${pair##*:}
    CUDA_VISIBLE_DEVICES=$gpu $M emotion/extract.py --manifest $NV/manifest_v2.jsonl --encoder $enc \
        --out $NV/feats > $LOG/nv2_extract_$enc.log 2>&1 &
    pids="$pids $!"
done
for p in $pids; do wait $p; done
grep -h "EXTRACT_DONE" $LOG/nv2_extract_*.log           # features made earlier are skipped, so no per-corpus line

for enc in whisper wavlm ast; do
    CUDA_VISIBLE_DEVICES=$G1 $M nonverbal/train.py --manifest $NV/manifest_v2.jsonl --feats $NV/feats --encoder $enc \
        --out $R/head_$enc.json --save-model $MD/nv_heads/$enc.pt --save-probs $R/head_${enc}_test.npz \
        > $LOG/nv2_train_$enc.log 2>&1
    grep -E "test all|TRAIN_DONE" $LOG/nv2_train_$enc.log
done

CUDA_VISIBLE_DEVICES=$G1 $M nonverbal/eval_gaps.py --candidates $R/nonverbaltts_gaps.json --models whisper wavlm ast \
    --out $R/nonverbaltts_gaps_heads.json > $LOG/nv2_eval_gaps.log 2>&1
grep -E "^==|^   " $LOG/nv2_eval_gaps.log

CUDA_VISIBLE_DEVICES=$G1 $M nonverbal/relabel.py --work $MELATI_DATA/bench/pf_test/work_rel \
    --sources $MELATI_DATA/bench/pf_test/sources.jsonl --models whisper wavlm ast \
    --thresholds $R/nonverbaltts_gaps_heads.json --new-work $MELATI_DATA/bench/pf_test/work_nvheads \
    --out $F/podcastfillers_nvheads.json > $LOG/nv2_rescore_pf.log 2>&1
tail -2 $LOG/nv2_rescore_pf.log
echo NV_HEADS2_DONE
