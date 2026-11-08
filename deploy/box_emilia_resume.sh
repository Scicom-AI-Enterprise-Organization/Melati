#!/bin/bash
# Finish an upstream Emilia-Pipe baseline that crashed part-way, with upstream's code unchanged.
#
#   bash deploy/box_emilia_resume.sh <gpu> <dir>
#
# Upstream's main loop has no per-file error handling, so one recording that yields no segments ends
# the whole batch (utils/tool.py filter(): ZeroDivisionError on 0 segments; docs/emilia_bugs.md).
# This reruns upstream on the recordings with no output yet, from a fresh folder of symlinks; when a
# run dies, the recording it was on is recorded as producing nothing (an empty segment list, which
# is what upstream's output for it amounts to: no data, and a stopped batch) and the rest go again.
# Outputs are moved into <dir>/emilia_in_processed/, where scripts/compare_emilia.py reads them.
# Every attempt's wall time is appended to <dir>/emilia_timing_resume.jsonl.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU=${1:?gpu}
P=${2:?dir holding emilia_in/ and emilia_in_processed/}
OUT=$P/emilia_in_processed
REST=$P/emilia_rest
NVLIB=$(ls -d /emilia/emilia/lib/python3.10/site-packages/nvidia/*/lib | paste -sd: -)
export HF_HOME=$MELATI_DATA/hf
cd "$MELATI_REPO/baselines/emilia_upstream"
for attempt in $(seq 1 30); do
  rm -rf "$REST" "${REST}_processed"; mkdir -p "$REST"
  n=0
  for f in "$P"/emilia_in/*; do
    id=$(basename "${f%.*}")
    [ -s "$OUT/$id/$id.json" ] && continue
    ln -s "$(readlink -f "$f")" "$REST/$(basename "$f")"; n=$((n + 1))
  done
  echo "=== attempt $attempt: $n recordings left $(date +%T)"
  [ "$n" = 0 ] && break
  sed "s|\"input_folder_path\": \"[^\"]*\"|\"input_folder_path\": \"$REST\"|" "$P/emilia_config.json" > "$P/emilia_config_rest.json"
  T0=$(date +%s)
  env CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH:-}" \
      /emilia/emilia/bin/python main.py --config_path "$P/emilia_config_rest.json" --batch_size 16 \
      --compute_type float16 --whisper_arch medium > "$P/emilia_run_resume$attempt.log" 2>&1
  EC=$?
  T1=$(date +%s)
  done_now=$(find "${REST}_processed" -mindepth 2 -maxdepth 2 -name "*.json" 2>/dev/null | wc -l)
  echo "{\"attempt\": $attempt, \"wall_s\": $((T1 - T0)), \"recordings\": $n, \"finished\": $done_now, \"exit\": $EC}" \
      >> "$P/emilia_timing_resume.jsonl"
  for d in "${REST}_processed"/*/; do
    [ -d "$d" ] || continue
    id=$(basename "$d")
    if [ -s "$d/$id.json" ]; then rm -rf "$OUT/$id"; mv "$d" "$OUT/$id"; fi
  done
  if [ "$EC" != 0 ]; then
    # the recording upstream was on when it died: the last "Processing audio" line
    bad=$(grep -o "Processing audio: [^,]*" "$P/emilia_run_resume$attempt.log" | tail -1 | sed 's/Processing audio: //')
    err=$(grep -E "Error|Exception" "$P/emilia_run_resume$attempt.log" | grep -v onnxruntime | tail -1 | cut -c1-200)
    [ -z "$bad" ] && { echo "!! died before any recording: $err"; break; }
    echo "!! upstream died on $bad: $err"
    mkdir -p "$OUT/$bad"; echo "[]" > "$OUT/$bad/$bad.json"
    echo "{\"recording\": \"$bad\", \"error\": \"$(echo "$err" | tr -d '"\\')\"}" >> "$P/emilia_crashed.jsonl"
  fi
done
echo "finished: $(find "$OUT" -mindepth 2 -maxdepth 2 -name '*.json' | wc -l) of $(ls "$P/emilia_in" | wc -l); crashed: $(cat "$P/emilia_crashed.jsonl" 2>/dev/null | wc -l)"
echo EMILIA_RESUME_DONE
