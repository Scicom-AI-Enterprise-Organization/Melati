#!/bin/bash
# The aligner service's venv. Where-Words-Begin-and-End was released from an environment that pins
# "transformers>=4.57" with no ceiling, i.e. 5.x: its tokenizer_config.json stores extra_special_tokens as a
# list, which 4.57 cannot read ("'list' object has no attribute 'keys'"). Each service has its own venv, so
# this one simply runs what the model was saved with: 5.20.2, the last 5.x before the dataset's run began.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export UV_CACHE_DIR=$MELATI_DATA/uv-cache UV_HTTP_TIMEOUT=600
V=$MELATI_ROOT/.venv-align
[ -d "$V" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
uv pip install --python "$V/bin/python" "${MIRROR[@]}" "transformers==5.20.2" numpy soundfile huggingface_hub fastapi \
  "uvicorn[standard]" 2>&1 | tail -1
"$V/bin/python" -c "import torch, transformers; print('align venv: torch', torch.__version__, 'transformers', transformers.__version__)"
echo ALIGN_SETUP_DONE
