#!/bin/bash
# rel5: the other way out of the licence conflict. NonverbalTTS's labels (CC BY-NC-SA 4.0) and PodcastFillers' (Adobe,
# non-commercial research, its licence travelling with derivatives) cannot both hold for one set of weights, and every
# head without NonverbalTTS (rel2nonv, rel3, rel4) collapsed on its test split (thresholds at 0.95, a few percent of the
# breaths and laughs). So rel5 keeps NonverbalTTS and drops PodcastFillers entirely: NonverbalTTS's candidates, AMI
# (marked candidates, unmarked ones as `other`, and word cuts: fillers and word tails), COUGHVID and Coswara, all of
# which a CC BY-NC-SA 4.0 release can carry. PodcastFillers' test episodes then score a head that never saw a podcast
# of theirs.
#
#   bash deploy/box_nv_rel5.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
MD=$MELATI_DATA/models
LOG=$MELATI_LOGS
RULE='{"floor_db": 2}'
V=${2:-rel5}

[ -s $NV/manifest_amiw.jsonl ] || $M nonverbal/candidates.py ami --words 20 --corpus ami_words \
    --out $NV/manifest_amiw.jsonl > $LOG/nvrel5_amiw.log 2>&1
tail -1 $LOG/nvrel5_amiw.log
$M emotion/extract.py --manifest $NV/manifest_amiw.jsonl --encoder whisper --out $NV/feats > $LOG/nvrel5_extract.log 2>&1
grep -E "EXTRACT_DONE" $LOG/nvrel5_extract.log

$M - $NV <<'PY'
import json, sys
from collections import Counter
nv = sys.argv[1]
rd = lambda f: [json.loads(l) for l in open(f"{nv}/{f}.jsonl", encoding="utf-8")]
rows = [r for r in rd("manifest_rel2") if r["corpus"].startswith("nvtts")]
rows += [dict(r, pool="coughs") for r in rd("manifest_rel2") if r["corpus"] == "coughvid"] + rd("manifest_coswara")
rows += [dict(r, pool="ami") for f in ("manifest_amic", "manifest_amiu", "manifest_amiw") for r in rd(f)]
assert not any(r["corpus"].startswith("p") for r in rows)          # nothing of PodcastFillers
with open(f"{nv}/manifest_rel5.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rel5 train", dict(Counter(f"{r.get('pool') or r['corpus']}/{r['label']}" for r in rows if r["split"] == "train")))
PY

mkdir -p $MD/nv_heads_$V
$M nonverbal/train.py --manifest $NV/manifest_rel5.jsonl --feats $NV/feats --encoder whisper \
    --out $R/head_whisper_$V.json --save-model $MD/nv_heads_$V/whisper.pt > $LOG/nv${V}_train.log 2>&1
grep -E "test |TRAIN_DONE" $LOG/nv${V}_train.log
MELATI_NV_HEADS=$MD/nv_heads_$V $M nonverbal/gap_rule.py eval --model whisper --rules "$RULE" \
    --out $R/gap_eval_$V.json > $LOG/nv${V}_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nv${V}_eval.log | tail -1
$M nonverbal/eval_ami.py --heads $MD/nv_heads_$V --thresholds $R/gap_eval_$V.json --out $R/ami_eval_$V.json \
    > $LOG/nv${V}_ami.log 2>&1
grep -vE "warn|Warn|^  " $LOG/nv${V}_ami.log | tail -3
echo NV_REL5_DONE
