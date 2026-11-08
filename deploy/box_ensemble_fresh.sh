#!/bin/bash
# Melati-AudioSet-Ensemble loaded the way someone else would load it: a new venv holding only the card's pip line at
# the newest versions pip resolves, an empty Hugging Face cache and no token, then each member against the
# probabilities evaluate.py cached and the card's Use snippet as written (audioset/check_ensemble_fresh.py). NAME labels
# the venv and its cache; PKGS replaces the packages (default: the card's five); GPU picks the GPU, and GPU= (empty)
# runs on the CPU, best with N=8; REVISION pins the repo's commit; RESULT is where the result goes (default
# $MELATI_DATA/tmp/fresh_$NAME.json); FLAGS passes --no-snippet or --snippet-only; REUSE=1 keeps the venv and cache
# of an earlier run of the same NAME. Exits with the check's own status.
#
#   NAME=minimal bash deploy/box_ensemble_fresh.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
NAME=${NAME:-minimal}
V=$MELATI_DATA/venvs/ens-$NAME
FX=$MELATI_DATA/tmp/ensemble_fixture.npz
[ -s $FX ] || $MELATI_ROOT/.venv-sed/bin/python audioset/ensemble_fixture.py --n 64 --out $FX || exit 1
export HF_HOME=$MELATI_DATA/hf-fresh-$NAME HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
if [ "${REUSE:-0}" != 1 ] || [ ! -x "$V/bin/python" ]; then
    rm -rf "$V" "$HF_HOME"
    uv venv -q --python "${PY:-3.12}" "$V" && uv pip install -q --python "$V/bin/python" \
        ${PKGS:-torch numpy huggingface_hub safetensors soundfile} || { echo FRESH_INSTALL_FAILED; exit 1; }
fi
CUDA_VISIBLE_DEVICES=${GPU-4} "$V/bin/python" -u audioset/check_ensemble_fresh.py --fixture $FX ${N:+--n $N} \
    ${REVISION:+--revision $REVISION} --out "${RESULT:-$MELATI_DATA/tmp/fresh_$NAME.json}" ${FLAGS:-}
rc=$?
echo "FRESH_DONE $rc"
exit $rc
