#!/bin/bash
# Voice-free arms with Whisper's token suppression on (Hugging Face's default) and off (as vLLM decodes),
# for a few checkpoints: how much of "hallucination" is the decoder being forbidden to stop.
#
#   bash deploy/box_asr_suppress_check.sh <gpu> <name=model> [...]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}; shift
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled
PY=$MELATI_ROOT/.venv-asr/bin/python
OUT=$MELATI_DATA/asr/suppress_check
for spec in "$@"; do
  name=${spec%%=*}; model=${spec#*=}
  for mode in ${MODES:-vllm hf}; do
    flag=""; [ "$mode" = hf ] && flag="--hf-suppress"
    CUDA_VISIBLE_DEVICES=$GPU $PY bench/run_benchmark.py --model "$model" --device cuda:0 --batch-size ${BS:-24} --overwrite \
        --arms silence music nonspeech $flag --out "$OUT/$name/$mode" > "$OUT.$name.$mode.log" 2>&1
    $PY - "$OUT/$name/$mode" "$name" "$mode" <<'PY'
import json, glob, sys
sys.path.insert(0, "bench")
from metrics import normalise
d, name, mode = sys.argv[1:]
row = []
for arm in ("silence", "music", "nonspeech"):
    f = glob.glob(f"{d}/*/{arm}.jsonl")[0]
    rs = [json.loads(l) for l in open(f)]
    row.append(f"{arm} {sum(bool(normalise(r['hyp'])) for r in rs) / len(rs):.3f}")
print(f"{name:12s} {mode:5s}", "  ".join(row), flush=True)
PY
  done
done
echo SUPCHECK_DONE
