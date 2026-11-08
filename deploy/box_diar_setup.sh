#!/bin/bash
# Melati's diarization venv: the vLLM nightly that registers MOSS-Transcribe-Diarize, pinned to the same
# commit the Who-Spoke-When-Over-the-Phone release was built and evaluated on (Speaker-Diarization,
# deploy/box_vllm_setup.sh). vLLM 0.17.1, which serves Whisper, refuses the architecture.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export UV_CACHE_DIR=$MELATI_DATA/uv-cache UV_HTTP_TIMEOUT=600   # 30 s timed out under the corpus downloads
V=$MELATI_ROOT/.venv-diar
[ -d "$V" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" -U vllm --torch-backend=auto \
  "${MIRROR[@]}" --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129 \
  --index-strategy unsafe-best-match 2>&1 | tail -3
uv pip install --python "$V/bin/python" "${MIRROR[@]}" soundfile soxr numpy 2>&1 | tail -1
"$V/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo DIAR_SETUP_DONE
