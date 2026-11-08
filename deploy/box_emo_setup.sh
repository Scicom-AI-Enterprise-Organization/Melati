#!/bin/bash
# The venv Melati's emotion heads are trained, scored and served in: .venv-melati
# with deploy/box_melati_setup.sh's pins (torch 2.8 + transformers 4.57.1, FunASR for emotion2vec+), only the packages
# the emotion work needs; box_melati_setup.sh adds the rest of the pipeline to the same venv later. Everything goes
# under $MELATI_ROOT and $MELATI_DATA (deploy/env.sh).
#
#   bash deploy/box_emo_setup.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
R=$MELATI_ROOT
export UV_CACHE_DIR=$MELATI_DATA/uv-cache
mkdir -p "$MELATI_DATA"

V=$R/.venv-melati
[ -d "$V" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0" \
  --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
uv pip install --python "$V/bin/python" "${MIRROR[@]}" "transformers==4.57.1" "numpy>=1.26,<2.3" scipy pandas \
  scikit-learn soundfile soxr pyarrow "huggingface_hub[hf_transfer]" einops safetensors httpx requests aiohttp tqdm \
  pyyaml librosa timm omegaconf "peft==0.17.1" pytest 2>&1 | tail -1
# emotion2vec+ runs through FunASR; the pins keep it from moving torch or transformers underneath
printf 'torch==2.8.0\ntorchaudio==2.8.0\ntransformers==4.57.1\nnumpy<2.3\n' > /tmp/melati_pins.txt
uv pip install --python "$V/bin/python" "${MIRROR[@]}" -c /tmp/melati_pins.txt funasr modelscope "datasets==4.1.1" \
  2>&1 | tail -1

"$V/bin/python" - <<'PY'
import funasr, torch, transformers
print("melati venv: torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__,
      "funasr", funasr.__version__)
PY
echo EMO_SETUP_DONE
