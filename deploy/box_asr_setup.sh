#!/bin/bash
# The ASR venv: training (transformers + peft) and the benchmark harness share it.
#
#   bash deploy/box_asr_setup.sh
#
# Venvs live on the local disk ($MELATI_ROOT), data on NFS ($MELATI_DATA): a venv on NFS
# makes every import a network round trip.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
V=$MELATI_ROOT/.venv-asr
export UV_CACHE_DIR=$MELATI_DATA/uv-cache
[ -d "$V" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" \
  "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
uv pip install --python "$V/bin/python" \
  "transformers==4.57.1" "peft==0.17.1" "accelerate==1.10.1" "datasets==4.1.1" \
  "huggingface_hub[hf_transfer]" soundfile librosa soxr scipy pyarrow requests wandb pandas 2>&1 | tail -2
"$V/bin/python" - <<'PY'
import torch, transformers, peft, datasets, soundfile
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
print("libsndfile", soundfile.__libsndfile_version__)
PY
echo ASR_SETUP_DONE
