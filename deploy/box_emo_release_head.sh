#!/bin/bash
# The emotion head that can be released: Whisper-large-v3, trained only on corpora whose licence explicitly allows
# derivatives under non-commercial research terms (CC0, CC BY, CC BY-NC, MIT, Apache-2.0, EmoV-DB's non-commercial
# licence): EMNS, Emozionalmente, JL-Corpus, MESD, Oreau, RESD, SUBESCO, EmoV-DB, Expresso. ShareAlike (CREMA-D,
# RAVDESS, CaFE, PAVOQUE, MELD, NonverbalTTS's labels), NoDerivatives (TurEV-DB) and corpora with no licence text
# are left out. Held out as before (nEMO, NonverbalTTS's VoxCeleb half), which is evaluation, not redistribution.
#
#   bash deploy/box_emo_release_head.sh 4
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=${1:-4}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=$G
A=$MELATI_ROOT/.venv-asr/bin/python
E=$MELATI_DATA/emo
O=$MELATI_DATA/results/emotion/release
H=$MELATI_DATA/models/ser_heads_release
mkdir -p $O $H
$A - $E/manifest.jsonl $E/manifest_release.jsonl <<'PY'
import json, sys
from collections import Counter
CLEAN = {"emns", "emozionalmente", "jl_corpus", "mesd", "oreau", "resd", "subesco", "emovdb", "expresso"}
HELD = {"nemo", "nvtts_voxceleb"}
kept = Counter()
with open(sys.argv[2], "w", encoding="utf-8") as out:
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        if r["corpus"] in CLEAN or r["corpus"] in HELD:
            kept[r["corpus"]] += 1
            out.write(line)
print("kept:", dict(sorted(kept.items())), "train clips:", sum(v for k, v in kept.items() if k in CLEAN))
PY
$A emotion/probe.py --manifest $E/manifest_release.jsonl --feats $E/feats --encoder whisper --out $O/probe_whisper.json \
    --save-probs $O/probs_whisper.npz --save-model $H/whisper.pt 2>&1 | grep -E "loco|held out|loco_mean" | cut -c1-240
echo EMO_REL_DONE
