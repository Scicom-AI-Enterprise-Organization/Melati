#!/bin/bash
# One vLLM Whisper server per GPU, OpenAI-compatible (/v1/audio/transcriptions).
#
#   bash serve/whisper_vllm.sh <model dir or hub id> <gpu,gpu,...> [first port]
#   bash serve/whisper_vllm.sh $MELATI_DATA/asr/runs/<best>/merged 0,1 8100
#
# Every server is named `melati-asr`, so a client never needs to know which checkpoint it hits.
# Logs go to $MELATI_DATA/logs/vllm-asr-<port>.log; a server is up when /health answers.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../deploy/env.sh"
MODEL=${1:?model}
GPUS=${2:?gpus}
PORT=${3:-8100}
export HF_HOME=$MELATI_DATA/hf VLLM_CONFIGURE_LOGGING=1
mkdir -p $MELATI_DATA/logs
for g in ${GPUS//,/ }; do
  CUDA_VISIBLE_DEVICES=$g setsid $MELATI_ROOT/.venv-vllm/bin/vllm serve "$MODEL" \
      --served-model-name melati-asr --port "$PORT" --host 127.0.0.1 \
      --gpu-memory-utilization 0.85 --max-num-seqs 256 --dtype bfloat16 \
      > $MELATI_DATA/logs/vllm-asr-$PORT.log 2>&1 < /dev/null &
  echo "gpu $g -> http://127.0.0.1:$PORT (pid $!)"
  PORT=$((PORT + 1))
done
