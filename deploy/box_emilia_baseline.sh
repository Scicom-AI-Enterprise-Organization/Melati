#!/bin/bash
# Upstream Emilia-Pipe (Amphion, vendored in baselines/emilia_upstream) on a set of recordings.
#
#   bash deploy/box_emilia_baseline.sh <gpu> <dir>
#
# <dir> holds the recordings to process in <dir>/emilia_in/ (scripts/make_pilot.py writes it;
# for VoxConverse it is a folder of symlinks to the WAVs). Upstream writes its segments to
# <dir>/emilia_in_processed/<id>/, and this writes <dir>/emilia_timing.json (wall seconds), which
# scripts/compare_emilia.py reads.
#
# The code is upstream's, unchanged; only the config is ours, and only where upstream's own
# README says to fill it in: the input folder, the two ONNX models (the copies the Scicom fork
# already keeps in /emilia), the HF token, and the language list, which must name Malay or every
# Malay segment is dropped. Whisper is upstream's default, `medium`, as in the Emilia paper.
# It runs in the Scicom fork's venv (/emilia/emilia: pyannote 3.1, WhisperX, onnxruntime-gpu),
# whose install order its CLAUDE.md documents; nothing in that venv is changed.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}
P=${2:?dir holding emilia_in/}
CFG=$P/emilia_config.json
cat > "$CFG" <<EOF
{
  "language": {"multilingual": true, "supported": ["ms", "en", "zh"]},
  "entrypoint": {"input_folder_path": "$P/emilia_in", "SAMPLE_RATE": 24000},
  "separate": {"step1": {"model_path": "/emilia/UVR-MDX-NET-Inst_HQ_3.onnx", "denoise": true,
               "margin": 44100, "chunks": 15, "n_fft": 6144, "dim_t": 8, "dim_f": 3072}},
  "mos_model": {"primary_model_path": "/emilia/sig_bak_ovr.onnx"},
  "huggingface_token": "$HF_TOKEN"
}
EOF
# every CUDA library the venv's wheels carry: ONNX Runtime's CUDA provider needs nvrtc, cufft and
# the runtime besides cuDNN/cuBLAS, and when one is missing it falls back to the CPU with only an
# error line in the log, so UVR runs there and the throughput number measures nothing
NVLIB=$(ls -d /emilia/emilia/lib/python3.10/site-packages/nvidia/*/lib | paste -sd: -)
cd "$MELATI_REPO/baselines/emilia_upstream"
export HF_HOME=$MELATI_DATA/hf
N=$(find -L "$P/emilia_in" -maxdepth 1 -type f | wc -l)
T0=$(date +%s)
env CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH:-}" \
    /emilia/emilia/bin/python main.py --config_path "$CFG" --batch_size 16 --compute_type float16 \
    --whisper_arch medium > "$P/emilia_run.log" 2>&1
EC=$?
T1=$(date +%s)
if grep -q "Failed to load library libonnxruntime_providers_cuda" "$P/emilia_run.log"; then
  echo "!! ONNX Runtime fell back to the CPU: the timing is not upstream's"; EC=99
fi
DONE=$(find "$P/emilia_in_processed" -mindepth 2 -maxdepth 2 -name "*.json" 2>/dev/null | wc -l)
echo "{\"wall_s\": $((T1 - T0)), \"gpu\": \"$GPU\", \"recordings\": $N, \"finished\": $DONE, \"exit\": $EC}" > "$P/emilia_timing.json"
grep -vE "^\s*$" "$P/emilia_run.log" | grep -E "Error|Traceback|All done" | tail -12
cat "$P/emilia_timing.json"
echo EMILIA_BASELINE_DONE
