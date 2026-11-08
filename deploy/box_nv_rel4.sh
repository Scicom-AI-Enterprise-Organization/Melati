#!/bin/bash
# rel4: rel3 plus AMI's unmarked gap candidates as `other`, the role NonverbalTTS's untagged candidates played: what
# a gap holds when it holds nothing worth a tag (word tails, clicks, room noise). Without NonverbalTTS, rel2nonv had to
# raise its thresholds to 0.95 to reach 80% precision there, and found 2% of the breaths and 4% of the laughs.
# AMI marks no breathing, so an unmarked candidate that a licence-clean head (rel2nonv: PodcastFillers' CC BY breaths)
# calls a breath at p >= 0.5 is left out rather than taught as `other`. AMI's marked and unmarked candidates are one
# pool in the loss, as AMI is one corpus.
#
#   bash deploy/box_nv_rel4.sh 4
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
V=${2:-rel4}

[ -s $NV/manifest_amiu.jsonl ] || $M nonverbal/candidates.py ami --unmarked other --only-unmarked \
    --corpus ami_unmarked --out $NV/manifest_amiu.jsonl > $LOG/nvrel4_amiu.log 2>&1
tail -1 $LOG/nvrel4_amiu.log
$M emotion/extract.py --manifest $NV/manifest_amiu.jsonl --encoder whisper --out $NV/feats > $LOG/nvrel4_extract.log 2>&1
grep -E "EXTRACT_DONE" $LOG/nvrel4_extract.log

$M - $NV $MD/nv_heads_rel2nonv/whisper.pt <<'PY'
import json, sys
from collections import Counter
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "emotion")
from probe import load
from melati.nv_gaps import HEAD_LABELS
from melati.ser import head_module
nv, ck = sys.argv[1], torch.load(sys.argv[2], map_location="cpu")
X, rows = load(f"{nv}/manifest_amiu.jsonl", f"{nv}/feats", "whisper")
head = head_module(ck["layers"], ck["dim"], n=len(HEAD_LABELS))
head.load_state_dict(ck["state"]); head.eval()
with torch.no_grad():
    p = torch.cat([head(torch.from_numpy(np.asarray(X[s:s + 4096])).float()).softmax(-1)
                   for s in range(0, len(X), 4096)]).numpy()
top = Counter(HEAD_LABELS[k] for k in p.argmax(1))
breath = p[:, HEAD_LABELS.index("breath")]
keep = [dict(r, pool="ami") for r, b in zip(rows, breath) if b < 0.5]
print(f"unmarked AMI candidates: {len(rows)}, most probable class {dict(top.most_common())}; "
      f"{len(rows) - len(keep)} called breath at p >= 0.5 and left out", flush=True)
base = [json.loads(l) for l in open(f"{nv}/manifest_rel3.jsonl", encoding="utf-8")]
base = [dict(r, pool="ami") if r["corpus"] == "ami_cands" else r for r in base]
with open(f"{nv}/manifest_rel4.jsonl", "w", encoding="utf-8") as fh:
    for r in base + keep:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rel4 train", dict(Counter(f"{r.get('pool') or r['corpus']}/{r['label']}" for r in base + keep
                                 if r["split"] == "train")), flush=True)
PY

mkdir -p $MD/nv_heads_$V
$M nonverbal/train.py --manifest $NV/manifest_rel4.jsonl --feats $NV/feats --encoder whisper \
    --out $R/head_whisper_$V.json --save-model $MD/nv_heads_$V/whisper.pt > $LOG/nv${V}_train.log 2>&1
grep -E "test |TRAIN_DONE" $LOG/nv${V}_train.log
MELATI_NV_HEADS=$MD/nv_heads_$V $M nonverbal/gap_rule.py eval --model whisper --rules "$RULE" \
    --out $R/gap_eval_$V.json > $LOG/nv${V}_eval.log 2>&1
grep -vE "warn|Warn" $LOG/nv${V}_eval.log | tail -1
$M nonverbal/eval_ami.py --heads $MD/nv_heads_$V --thresholds $R/gap_eval_$V.json --out $R/ami_eval_$V.json \
    > $LOG/nv${V}_ami.log 2>&1
grep -vE "warn|Warn|^  " $LOG/nv${V}_ami.log | tail -3
echo NV_REL4_DONE
