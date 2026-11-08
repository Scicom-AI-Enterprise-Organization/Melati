#!/bin/bash
# A new version of Melati-AudioSet-Ensemble, which is public (audioset/release_ensemble.py --stage, --check-staging,
# --merge): the code staged on a branch; loaded from there in a new venv holding only the card's pip lines, into an
# empty cache and with no token, the ensemble and its engine both (deploy/box_ensemble_fresh.sh, its result on the
# card); the engine from the Hub measured on the evaluation set, on its own and over HTTP (audioset/check_engine.py,
# ENGINE_GPUS); the ensemble checked from the Hub on the whole evaluation set in fp32 and bf16 and carded there; the
# card's snippet run in the new venv as well; and only then put on main. MESSAGE is the commit message; EVAL_GPU the
# GPU of the single-GPU steps (default 1).
#
#   MESSAGE='...' bash deploy/box_audioset_ensemble_update.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf OMP_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
GPU=${EVAL_GPU:-1}
GPUS=${ENGINE_GPUS:-0,2,4,5}
R=("$MELATI_ROOT/.venv-sed/bin/python" audioset/release_ensemble.py --out "$MELATI_DATA/release/Melati-AudioSet-Ensemble"
   --repo Scicom-intl/Melati-AudioSet-Ensemble --message "$MESSAGE")
LOG=$MELATI_DATA/tmp/asupd_stage.log
CUDA_VISIBLE_DEVICES=$GPU "${R[@]}" --stage 2>&1 | tee $LOG
REV=$(grep -o "STAGED [0-9a-f]*" $LOG | cut -d" " -f2)
[ -n "$REV" ] || { echo "ASUPD_FAILED: nothing staged"; exit 1; }
NAME=minimal GPU=$GPU REVISION=$REV RESULT=$MELATI_DATA/results/audioset/ensemble_fresh.json \
    PKGS="torch numpy huggingface_hub safetensors soundfile fastapi uvicorn" FLAGS="--no-snippet --engine" \
    bash deploy/box_ensemble_fresh.sh
CUDA_VISIBLE_DEVICES=$GPUS $MELATI_ROOT/.venv-sed/bin/python -u audioset/check_engine.py --revision $REV \
    --gpus "$(seq -s, 0 $(( $(tr , '\n' <<< "$GPUS" | wc -l) - 1 )))" --http
CUDA_VISIBLE_DEVICES=$GPU "${R[@]}" --check-staging
NAME=minimal GPU=$GPU REVISION=staging REUSE=1 FLAGS=--snippet-only bash deploy/box_ensemble_fresh.sh
CUDA_VISIBLE_DEVICES=$GPU "${R[@]}" --merge
echo ASUPD_DONE
