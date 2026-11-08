#!/bin/bash
# Melati's five emotion heads again (melati/ser.py: h_whisper, h_whisper_ft, h_w2vbert, h_wavlm, h_e2v), after the box
# that held them was wiped on 2026-10-29. The same 21 training corpora and 2 held-out ones the first heads saw
# (results/emotion/manifest_stats.json, checked corpus by corpus and label by label before any head trains), every
# encoder's features, each head trained leave-one-corpus-out and in full (emotion/probe.py), MERaLiON-SER and
# emotion2vec+ scored on the same held-out clips, the seven-model average and its thresholds (emotion/ensemble.py),
# the served heads checked against the trained ones, and all of it set beside the first heads' numbers
# (emotion/compare_retrain.py). One deliberate change: h_whisper_ft reads the released ASR's encoder
# (melati.ser.WHISPER_FT), since the candidate the first one read is gone. Every step resumes from its own files.
#
# Needs deploy/box_emo_setup.sh, and the first heads' results (the repo's results/emotion) copied to
# $MELATI_DATA/v1/results/emotion. GPUS picks the GPUs (default "1 4"). ALLOW_DIFF=1 trains even when the manifest
# differs.
#
#   bash deploy/box_emo_retrain.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
D=$MELATI_DATA
export MELATI_DATA=$D HF_HOME=$D/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8
P=$MELATI_ROOT/.venv-melati/bin/python
E=$D/emo
O=$D/results/emotion
H=$D/models/ser_heads
V1=$D/v1/results/emotion
L=$MELATI_LOGS
GPUS=(${GPUS:-1 4})
ENC=(whisper whisper_ft w2vbert wavlm e2v)
mkdir -p $E $O $H $L
[ -s $V1/manifest_stats.json ] || { echo "the first heads' results are not at $V1"; exit 1; }

# each GPU takes every n-th encoder and runs its share in turn: on_gpus <function>
on_gpus() {
    local pids=() failed=0 g k
    for g in $(seq 0 $((${#GPUS[@]} - 1))); do
        ( for k in $(seq $g ${#GPUS[@]} $((${#ENC[@]} - 1))); do $1 ${GPUS[$g]} ${ENC[$k]} || exit 1; done ) &
        pids+=($!)
    done
    for g in "${pids[@]}"; do wait "$g" || failed=$((failed + 1)); done
    return $failed
}

# 1. the ten sources the first heads' corpora came from (TESS, AESDD, EMOVO and the Polish set were in no head)
echo "[emo] fetch"
$P -u emotion/fetch_corpora.py --root $E/raw --only cameo nonverbaltts expresso meld emovdb emodb shemo ased turevdb \
    urdu > $L/emo_fetch.log 2>&1
if grep -q FAILED $E/raw/fetch_status.json; then grep FAILED $E/raw/fetch_status.json; echo EMO_FETCH_FAILED; exit 1; fi

# 2. the manifest, and every encoder's features of it
echo "[emo] manifest"
$P -u emotion/build_manifest.py --raw $E/raw --out $E/manifest.jsonl > $L/emo_manifest.log 2>&1
grep " clips, " $L/emo_manifest.log || true
extract() {
    CUDA_VISIBLE_DEVICES=$1 $P -u emotion/extract.py --manifest $E/manifest.jsonl --encoder $2 --out $E/feats \
        > $L/emo_extract_$2.log 2>&1 || { echo "EXTRACT_FAILED $2"; tail -5 $L/emo_extract_$2.log; return 1; }
    echo "[emo] features $2: $(grep -c 'clips in' $L/emo_extract_$2.log) corpora this pass"
}
on_gpus extract || { echo EMO_EXTRACT_FAILED; exit 1; }

# 3. what the heads will see must be what the first heads saw
if ! $P emotion/manifest_stats.py --manifest $E/manifest.jsonl --bad $E/feats/whisper --out $O/manifest_stats.json \
        --expect $V1/manifest_stats.json > $L/emo_stats.log 2>&1; then
    cat $L/emo_stats.log
    [ "${ALLOW_DIFF:-0}" = 1 ] || { echo EMO_MANIFEST_DIFFERS; exit 1; }
fi
tail -1 $L/emo_stats.log

# 4. the heads: leave one corpus out, then all training corpora with the held-out scores and thresholds
probe() {
    [ -s $H/$2.pt ] && [ -s $O/probe_$2.json ] && return 0
    CUDA_VISIBLE_DEVICES=$1 $P -u emotion/probe.py --manifest $E/manifest.jsonl --feats $E/feats --encoder $2 \
        --out $O/probe_$2.json --save-probs $O/probs_$2.npz --save-model $H/$2.pt > $L/emo_probe_$2.log 2>&1 \
        || { echo "PROBE_FAILED $2"; tail -5 $L/emo_probe_$2.log; return 1; }
    echo "[emo] head $2: $(grep -o '"held_out": {[^}]*' $L/emo_probe_$2.log | head -1 | cut -c1-120)"
}
on_gpus probe || { echo EMO_PROBE_FAILED; exit 1; }

# 5. the two released classifiers on the same held-out clips, the seven-model average, the served heads
echo "[emo] meralion + e2v"
[ -s $O/current.npz ] || CUDA_VISIBLE_DEVICES=${GPUS[0]} $P -u emotion/score_current.py --manifest $E/manifest.jsonl \
    --out $O/current.json > $L/emo_current.log 2>&1 || { tail -5 $L/emo_current.log; echo EMO_CURRENT_FAILED; exit 1; }
$P -u emotion/ensemble.py --members $O/probs_whisper.npz $O/probs_whisper_ft.npz $O/probs_w2vbert.npz \
    $O/probs_wavlm.npz $O/probs_e2v.npz $O/current.npz:meralion $O/current.npz:e2v --out $O/ensemble_all.json \
    > $O/ensemble_all.log 2>&1
CUDA_VISIBLE_DEVICES=${GPUS[0]} $P -u emotion/check_serving.py --manifest $E/manifest.jsonl --probs-dir $O --n 64 \
    > $L/emo_serving.log 2>&1 || { tail -5 $L/emo_serving.log; echo EMO_SERVING_FAILED; exit 1; }
grep "^h_" $L/emo_serving.log
$P emotion/compare_retrain.py --old $V1 --new $O --out $O/compare_v1.json
echo EMO_RETRAIN_DONE
