#!/bin/bash
# A variant of the licence-clean manifest (manifest_rel5.jsonl: NonverbalTTS, AMI, COUGHVID and Coswara), trained and
# scored on NonverbalTTS with the cough curve beside it: rows whose id matches DROP (a Python regex) are left out, and
# the pools weighed by WEIGHTS (train.py --pool-weights).
#
#   bash deploy/box_nv_variant.sh rel7 4 '{\"coughs\": 0.5}' 'cough-shallow'
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
V=$1
G=${2:-4}
W=${3:-'{}'}
DROP=${4:-'^$'}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'
$M - $NV $V "$DROP" <<'PY'
import json, re, sys
from collections import Counter
nv, v, drop = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
rows = [json.loads(l) for l in open(f"{nv}/manifest_rel5.jsonl", encoding="utf-8")]
kept = [r for r in rows if not drop.search(r["id"])]
with open(f"{nv}/manifest_{v}.jsonl", "w", encoding="utf-8") as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(v, "dropped", len(rows) - len(kept), "kept", dict(Counter(f"{r.get('pool') or r['corpus']}/{r['label']}"
                                                                 for r in kept if r["split"] == "train" and r["label"] == "cough")))
PY
mkdir -p $MD/nv_heads_$V
$M nonverbal/train.py --manifest $NV/manifest_$V.jsonl --feats $NV/feats --encoder whisper --pool-weights "$W" \
    --out $R/head_whisper_$V.json --save-model $MD/nv_heads_$V/whisper.pt > $LOG/nv${V}_train.log 2>&1
grep -E "TRAIN_DONE" $LOG/nv${V}_train.log
MELATI_NV_HEADS=$MD/nv_heads_$V $M nonverbal/gap_rule.py eval --model whisper --rules "$RULE" \
    --out $R/gap_eval_$V.json > $LOG/nv${V}_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nv${V}_eval.log | tail -1
$M scripts/gap_curve.py --heads $MD/nv_heads_$V --classes cough 2>&1 | grep -E "^calib|^test"
echo NV_VARIANT_DONE
