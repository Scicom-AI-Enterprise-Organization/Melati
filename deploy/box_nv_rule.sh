#!/bin/bash
# A looser gap finder for breaths, and the non-verbal head retrained on the candidates it finds.
#
#   bash deploy/box_nv_rule.sh 4
#
# nonverbal/gap_rule.py diagnose found the missed NonverbalTTS breaths sitting in real gaps, just under the noise-floor
# threshold (median 1.7 dB) or above it for under 80 ms. Training candidates are rebuilt here under the most permissive
# rule tried (RULE), the Whisper head retrained on them beside the hand-cut events (nv_heads_v3/, the live head is
# untouched), and every rule re-scored with it on NonverbalTTS, chosen on the calibration draw alone.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
RULE='{"floor_db": 2, "min_active": 0.04}'
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
H=$MELATI_DATA/models/nv_heads_v3
LOG=$MELATI_LOGS
mkdir -p $H
[ -s $NV/manifest_pfc2.jsonl ] || $M nonverbal/candidates.py pf --rule "$RULE" --corpus pf_cands2 \
    --out $NV/manifest_pfc2.jsonl > $LOG/nvrule_pfc.log 2>&1
[ -s $NV/manifest_nvc2.jsonl ] || $M nonverbal/candidates.py nvtts --rule "$RULE" --corpus nvtts_cands2 \
    --out $NV/manifest_nvc2.jsonl > $LOG/nvrule_nvc.log 2>&1
tail -1 $LOG/nvrule_pfc.log; tail -1 $LOG/nvrule_nvc.log
cat $NV/manifest_pfc2.jsonl $NV/manifest_nvc2.jsonl > $NV/manifest_cands2.jsonl
cat $NV/manifest.jsonl $NV/manifest_cands2.jsonl > $NV/manifest_v3.jsonl
$M emotion/extract.py --manifest $NV/manifest_cands2.jsonl --encoder whisper --out $NV/feats > $LOG/nvrule_extract.log 2>&1
grep -E "^\[|EXTRACT_DONE" $LOG/nvrule_extract.log
$M nonverbal/train.py --manifest $NV/manifest_v3.jsonl --feats $NV/feats --encoder whisper \
    --out $R/head_whisper_v3.json --save-model $H/whisper.pt > $LOG/nvrule_train.log 2>&1
grep -E "test all|TRAIN_DONE" $LOG/nvrule_train.log
MELATI_NV_HEADS=$H $M nonverbal/gap_rule.py eval --model whisper \
    --rules '{}' '{"floor_db": 3}' '{"floor_db": 2}' "$RULE" --out $R/gap_eval_v3head.json > $LOG/nvrule_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nvrule_eval.log | tail -5
echo NV_RULE_DONE
