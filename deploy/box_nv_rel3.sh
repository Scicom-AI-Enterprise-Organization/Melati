#!/bin/bash
# The released non-verbal head without NonverbalTTS. Its labels are CC BY-NC-SA 4.0 (the dataset's README; its
# metadata tag says apache-2.0), so they may calibrate and score a released head but not train one. In their place:
# AMI meeting candidates (CC BY 4.0; laughs, coughs and other vocal sounds between words, labelled by the
# transcribers' marks) and Coswara's coughs (CC BY 4.0), pooled with COUGHVID's. Two heads, so each change is
# measured on its own: rel2nonv (rel2 without NonverbalTTS: what dropping it costs) and rel3 (with AMI and Coswara).
# Every head, the shipped rel2 included, is scored on NonverbalTTS's test clips and on AMI's test meetings.
#
#   bash deploy/box_nv_rel3.sh 4 3
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
G2=${2:-3}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'

[ -s $NV/manifest_coswara.jsonl ] || $M nonverbal/build_manifest.py --coswara $NV/raw/coswara --out $NV \
    --name manifest_coswara > $LOG/nvrel3_coswara.log 2>&1
grep -E "coswara|MANIFEST_DONE" $LOG/nvrel3_coswara.log | tail -3
while [ ! -f $LOG/amifetch.done ]; do sleep 30; done          # nonverbal/ami.py fetch, started on its own
grep FETCH_DONE $LOG/amifetch.log
[ -s $NV/manifest_amic.jsonl ] || $M nonverbal/candidates.py ami --out $NV/manifest_amic.jsonl \
    > $LOG/nvrel3_ami.log 2>&1
tail -1 $LOG/nvrel3_ami.log
for m in manifest_coswara manifest_amic; do
    $M emotion/extract.py --manifest $NV/$m.jsonl --encoder whisper --out $NV/feats > $LOG/nvrel3_extract_$m.log 2>&1
    grep -E "EXTRACT_DONE" $LOG/nvrel3_extract_$m.log
done

# the shipped head on AMI, beside the training (another GPU)
CUDA_VISIBLE_DEVICES=$G2 $M nonverbal/eval_ami.py --heads $MD/nv_heads_rel2 --thresholds $R/gap_eval_rel2.json \
    --out $R/ami_eval_rel2.json > $LOG/nvrel3_ami_rel2.log 2>&1 &
BG=$!

$M - $NV <<'PY'
import json, sys
from collections import Counter
nv = sys.argv[1]
rows = [json.loads(l) for l in open(f"{nv}/manifest_rel2.jsonl", encoding="utf-8")]
rows = [dict(r, pool="coughs") if r["corpus"] == "coughvid" else r for r in rows if not r["corpus"].startswith("nvtts")]
extra = [json.loads(l) for f in ("manifest_coswara", "manifest_amic") for l in open(f"{nv}/{f}.jsonl", encoding="utf-8")]
for name, rs in (("rel2nonv", rows), ("rel3", rows + extra)):
    with open(f"{nv}/manifest_{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, dict(Counter(f"{r['corpus']}/{r['split']}/{r['label']}" for r in rs if r["split"] == "train")))
PY

for v in rel2nonv rel3; do
    mkdir -p $MD/nv_heads_$v
    $M nonverbal/train.py --manifest $NV/manifest_$v.jsonl --feats $NV/feats --encoder whisper \
        --out $R/head_whisper_$v.json --save-model $MD/nv_heads_$v/whisper.pt > $LOG/nv${v}_train.log 2>&1
    grep -E "test |TRAIN_DONE" $LOG/nv${v}_train.log
    MELATI_NV_HEADS=$MD/nv_heads_$v $M nonverbal/gap_rule.py eval --model whisper --rules "$RULE" \
        --out $R/gap_eval_$v.json > $LOG/nv${v}_eval.log 2>&1
    grep -vE "warn|Warn" $LOG/nv${v}_eval.log | tail -1
    $M nonverbal/eval_ami.py --heads $MD/nv_heads_$v --thresholds $R/gap_eval_$v.json --out $R/ami_eval_$v.json \
        > $LOG/nv${v}_ami.log 2>&1
    grep -vE "warn|Warn|^  " $LOG/nv${v}_ami.log | tail -3
done
wait $BG || echo "the shipped head's AMI eval failed"
grep -vE "warn|Warn|^  " $LOG/nvrel3_ami_rel2.log | tail -3
echo NV_REL3_DONE
