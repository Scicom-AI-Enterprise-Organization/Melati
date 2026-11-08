#!/bin/bash
# The released non-verbal head, licence-clean: PodcastFillers only from its CC BY 3.0 episodes (its other episodes are
# CC BY-SA or BY-ND, which cannot sit beside its non-commercial annotations in one set of weights), NonverbalTTS
# candidates, COUGHVID coughs (CC BY 4.0); no VocalSound (CC BY-SA). Scored with the shipped gap rule.
#
#   bash deploy/box_nv_release_head2.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
M=$MELATI_ROOT/.venv-melati/bin/python
NV=$MELATI_DATA/nv
R=$MELATI_DATA/results/nonverbal
H=$MELATI_DATA/models/nv_heads_rel2
LOG=$MELATI_LOGS
mkdir -p $H
$M - $NV/manifest_rel.jsonl $NV/pf_episode_licence.json $NV/manifest_rel2.jsonl <<'PY'
import json, sys
from collections import Counter
src, lic, dst = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
kept = Counter()
with open(dst, "w", encoding="utf-8") as out:
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        if r.get("episode") is not None and lic.get(r["episode"]) != "CC BY 3.0":
            continue
        kept[r["corpus"]] += 1
        out.write(line)
print("kept per corpus:", dict(kept))
PY
$M nonverbal/train.py --manifest $NV/manifest_rel2.jsonl --feats $NV/feats --encoder whisper \
    --out $R/head_whisper_rel2.json --save-model $H/whisper.pt > $LOG/nvrel2_train.log 2>&1
grep -E "test all|TRAIN_DONE" $LOG/nvrel2_train.log
MELATI_NV_HEADS=$H $M nonverbal/gap_rule.py eval --model whisper --rules '{"floor_db": 2}' \
    --out $R/gap_eval_rel2.json > $LOG/nvrel2_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nvrel2_eval.log | tail -2
echo NV_REL2_DONE
