#!/bin/bash
# The emotion heads again with MER2023 in training (Mandarin film and TV, 5,030 clips): a manifest with it added,
# its features on all five encoders (every other corpus's are reused), each head retrained leave-one-corpus-out and
# in full, and the seven-model average re-scored on the same held-out clips. Nothing served changes: the new heads
# land in ser_heads_mer2023/ and the results in results/emotion/mer2023/, beside the live ones.
#
#   bash deploy/box_emo_mer2023.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
A=$MELATI_ROOT/.venv-asr/bin/python          # what the live heads' features and probes ran in
M=$MELATI_ROOT/.venv-melati/bin/python       # emotion2vec (funasr) lives here
E=$MELATI_DATA/emo
R=$MELATI_DATA/results/emotion
O=$R/mer2023
H=$MELATI_DATA/models/ser_heads_mer2023
LOG=$MELATI_LOGS
mkdir -p $O $H

[ -s $E/raw/mer2023/.fetched ] || { echo "MER2023 is not extracted yet (emotion/fetch_mer2023.py)"; exit 1; }
$M emotion/build_manifest.py --raw $E/raw --out $E/manifest_mer2023.jsonl > $LOG/emomer_manifest.log 2>&1
# the new manifest must be the live one plus MER2023 and nothing else, or the comparison compares two things
$M - $E/manifest.jsonl $E/manifest_mer2023.jsonl <<'PY'
import json, sys
from collections import Counter
a, b = (Counter(json.loads(l)["corpus"] for l in open(f)) for f in sys.argv[1:3])
diff = {c: (a.get(c, 0), b.get(c, 0)) for c in set(a) | set(b) if a.get(c, 0) != b.get(c, 0)}
print("corpora that differ from the live manifest:", diff)
assert set(diff) == {"mer2023"}, diff
PY

for enc in whisper whisper_ft wavlm w2vbert; do
    $A emotion/extract.py --manifest $E/manifest_mer2023.jsonl --encoder $enc --out $E/feats > $LOG/emomer_extract_$enc.log 2>&1
done
$M emotion/extract.py --manifest $E/manifest_mer2023.jsonl --encoder e2v --out $E/feats > $LOG/emomer_extract_e2v.log 2>&1
grep -h "mer2023" $LOG/emomer_extract_*.log

for enc in whisper whisper_ft w2vbert wavlm e2v; do
    $A emotion/probe.py --manifest $E/manifest_mer2023.jsonl --feats $E/feats --encoder $enc --out $O/probe_$enc.json \
        --save-probs $O/probs_$enc.npz --save-model $H/$enc.pt > $LOG/emomer_probe_$enc.log 2>&1
    grep -E "loco mer2023|held out all|loco_mean" $LOG/emomer_probe_$enc.log | cut -c1-200
done
$A emotion/ensemble.py --members $O/probs_w2vbert.npz $O/probs_e2v.npz $O/probs_whisper.npz $O/probs_whisper_ft.npz \
    $O/probs_wavlm.npz $R/current.npz:meralion $R/current.npz:e2v --out $O/ensemble_all.json > $O/ensemble_all.log 2>&1
tail -3 $O/ensemble_all.log
echo EMOMER_DONE
