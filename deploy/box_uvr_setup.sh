#!/bin/bash
# The UVR-MDX-Net service's venv (comparisons only): torch for the STFT in upstream's ConvTDFNet, onnxruntime-gpu
# for the model, and the ONNX file Emilia's README downloads.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export UV_CACHE_DIR=$MELATI_DATA/uv-cache UV_HTTP_TIMEOUT=600
V=$MELATI_ROOT/.venv-uvr
[ -d "$V" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
uv pip install --python "$V/bin/python" "${MIRROR[@]}" "onnxruntime-gpu>=1.20" numpy librosa soundfile tqdm fastapi \
  "uvicorn[standard]" 2>&1 | tail -1
mkdir -p $MELATI_DATA/models/uvr
cp -n /emilia/UVR-MDX-NET-Inst_HQ_3.onnx $MELATI_DATA/models/uvr/ 2>/dev/null || \
  curl -sL -o $MELATI_DATA/models/uvr/UVR-MDX-NET-Inst_HQ_3.onnx \
  https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx
"$V/bin/python" -c "import onnxruntime as o, torch; print('uvr venv: ort', o.__version__, o.get_available_providers(), 'torch', torch.__version__)"
echo UVR_SETUP_DONE
