#!/bin/bash
# The pipeline venvs and model weights.
#
#   bash deploy/box_melati_setup.sh
#
#   .venv-melati  torch 2.8 + transformers 4.57: separation (DBS), aligner, SOTA-MOS, tagging,
#                 the stage runners
#   .venv-vllm    vLLM 0.17.1, which serves Whisper (the ASR)
#
# Third-party code is cloned under $MELATI_ROOT/ext and installed --no-deps, so it cannot move
# the pinned torch underneath the stages.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export UV_CACHE_DIR=$MELATI_DATA/uv-cache
export HF_HOME=$MELATI_DATA/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
EXT=$MELATI_ROOT/ext
M=$MELATI_DATA/models
mkdir -p "$EXT" "$M"

V=$MELATI_ROOT/.venv-melati
[ -d "$V" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0" \
  --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
uv pip install --python "$V/bin/python" "${MIRROR[@]}" "transformers==4.57.1" "numpy>=1.26,<2.3" scipy pandas scikit-learn \
  soundfile soxr pyloudnorm pyarrow "huggingface_hub[hf_transfer]" einops rotary-embedding-torch safetensors \
  fastapi "uvicorn[standard]" python-multipart httpx requests aiohttp tqdm pyyaml matplotlib onnxruntime \
  silero-vad librosa timm omegaconf "peft==0.17.1" wordfreq 2>&1 | tail -1

# emotion2vec+ runs through FunASR; the pins keep it from moving torch or transformers underneath
printf 'torch==2.8.0\ntorchaudio==2.8.0\ntransformers==4.57.1\nnumpy<2.3\n' > /tmp/melati_pins.txt
uv pip install --python "$V/bin/python" "${MIRROR[@]}" -c /tmp/melati_pins.txt funasr modelscope panns-inference \
  torchlibrosa "datasets==4.1.1" 2>&1 | tail -1

[ -d "$EXT/dbs" ] || git clone -q https://github.com/JusperLee/Dynamic-Band-Splitting "$EXT/dbs"
uv pip install --python "$V/bin/python" --no-deps -e "$EXT/dbs" 2>&1 | tail -1
[ -d "$EXT/SOTA-MOS" ] || git clone -q https://github.com/Scicom-AI-Enterprise-Organization/SOTA-MOS "$EXT/SOTA-MOS"
uv pip install --python "$V/bin/python" --no-deps -e "$EXT/SOTA-MOS" 2>&1 | tail -1
[ -d "$EXT/faster-UTMOSv2" ] || git clone -q https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2 "$EXT/faster-UTMOSv2"
uv pip install --python "$V/bin/python" --no-deps -e "$EXT/faster-UTMOSv2" 2>&1 | tail -1

W=$MELATI_ROOT/.venv-vllm
[ -d "$W" ] || uv venv --python 3.12 "$W"
uv pip install --python "$W/bin/python" "${MIRROR[@]}" "vllm==0.17.1" 2>&1 | tail -1
uv pip install --python "$W/bin/python" "${MIRROR[@]}" soundfile soxr librosa pyarrow "huggingface_hub[hf_transfer]" 2>&1 | tail -1

HF=$V/bin/hf
$HF download Scicom-intl/SOTA-MOS --include "model/*" --local-dir "$M/sotamos" 2>&1 | tail -1
# the released ASR at the revision serve/cluster.json serves
$HF download Scicom-intl/Melati-Whisper-Large-V3 --revision cdbb981c6ce8d87b338ed9503ef9a51f24a2a6aa 2>&1 | tail -1
for repo in JusperLee/DBS-TIGER-DnR JusperLee/DBS-BSRNN-DnR \
            Scicom-intl/Where-Words-Begin-and-End-Whisper-Large-V3 \
            Scicom-intl/Who-Spoke-When-Over-the-Phone-Whisper-Large-V3-Merged \
            openai/whisper-large-v3 openai/whisper-medium; do
  $HF download "$repo" 2>&1 | tail -1
done
# PANNs Cnn14 (the front stage's sound events, melati/sed.py) and DNSMOS (serve/tag_server.py): about 330 MB
fetch() {                                      # $1 the file, $2 its source; a failed fetch leaves nothing behind
    [ -s "$1" ] && return
    mkdir -p "$(dirname "$1")"
    curl -fsSL --retry 3 "$2" -o "$1.part" && mv "$1.part" "$1"
}
fetch "$M/panns/Cnn14_DecisionLevelMax.pth" "https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth"
fetch "$M/panns/class_labels_indices.csv" \
    https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
fetch "$M/dnsmos/sig_bak_ovr.onnx" \
    https://github.com/microsoft/DNS-Challenge/raw/82f1b17e7776a43eee395d0f45bae8abb700ad00/DNSMOS/DNSMOS/sig_bak_ovr.onnx
# the non-verbal heads (nv_whisper, nvf_whisper), out of their public release (1.3 GB) into the files the tag server reads
if [ ! -s "$M/nv_heads/whisper.pt" ] || [ ! -s "$M/nv_frames/whisper.pt" ]; then
    "$V/bin/python" "$MELATI_REPO/nonverbal/from_release.py"
fi

"$V/bin/python" - <<'PY'
import torch, transformers, dbs, sotamos
print("melati venv: torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)
PY
"$W/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo MELATI_SETUP_DONE
