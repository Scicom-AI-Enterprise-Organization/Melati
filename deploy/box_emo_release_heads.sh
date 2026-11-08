#!/bin/bash
# The five emotion heads Melati's pipeline averages (melati/ser.py), public on the Hub as
# Scicom-intl/Melati-Emotion-Heads since 2026-11-08: the heads, the counts of what they trained on and their scores,
# and a card written from those scores (emotion/heads_card.py, which names every training corpus with its licence).
# No audio, no features. Read back from the Hub and checked byte for byte; deploy/box_melati_setup.sh downloads them
# at the revision melati/pins.py names.
#
#   bash deploy/box_emo_release_heads.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
D=$MELATI_DATA
export HF_HOME=$D/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
V=$MELATI_ROOT/.venv-melati/bin
REPO=Scicom-intl/Melati-Emotion-Heads
S=$D/release/emotion_heads
rm -rf "$S" "$D/release/emotion_heads_check"
mkdir -p "$S/results"
cp "$D"/models/ser_heads/{whisper,whisper_ft,w2vbert,wavlm,e2v}.pt "$S/"
for f in manifest_stats probe_whisper probe_whisper_ft probe_w2vbert probe_wavlm probe_e2v current ensemble_all \
         compare_v1; do
    cp "$D/results/emotion/$f.json" "$S/results/"
done
"$V/python" emotion/heads_card.py --results "$S/results" --out "$S/README.md"
"$V/hf" upload "$REPO" "$S" . --repo-type model --commit-message "Public: the five emotion heads, with their card" \
    > "$MELATI_LOGS/emoheads_upload.log" 2>&1
"$V/python" - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().update_repo_settings(sys.argv[1], private=False)
print(sys.argv[1], "is public at", HfApi().model_info(sys.argv[1]).sha)
PY
code=$(curl -s -o /dev/null -w '%{http_code}' "https://huggingface.co/api/models/$REPO")   # without a token
[ "$code" = 200 ] || { echo "NOT_PUBLIC $code"; exit 1; }
"$V/hf" download "$REPO" --repo-type model --local-dir "$D/release/emotion_heads_check" > /dev/null 2>&1
for f in whisper whisper_ft w2vbert wavlm e2v; do
    a=$(sha256sum "$S/$f.pt" | cut -d' ' -f1); b=$(sha256sum "$D/release/emotion_heads_check/$f.pt" | cut -d' ' -f1)
    [ "$a" = "$b" ] || { echo "RELEASE_DIFFERS $f"; exit 1; }
    echo "$f.pt ${a:0:12} read back the same"
done
echo EMO_HEADS_RELEASE_DONE
