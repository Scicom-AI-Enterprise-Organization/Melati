#!/bin/bash
# Melati v1: every recording of the podcast source (malaysia-ai/malaysian-podcast-youtube, 19,092 recordings, about
# 2,200 h) through the GPU stages, front to fill, on serve/cluster.json (GPUs 0-5) and the AudioSet tagger's own
# cluster (serve/cluster_sounds.json). The vocal sounds' AudioSet pass (deploy/box_audioset_pass.sh), export, the
# parquet shards and the Hugging Face upload run after it, at each publish (deploy/box_full_publish.sh).
#
#   bash deploy/box_full.sh
#
# K client processes each drive a 1/K slice of the source list: one asyncio loop also reads and slices whole
# recordings between its model calls, and at this cluster's size a single loop would be the bottleneck. Every stage
# resumes from its own files, so a pass re-runs only what is missing or failed. A watchdog runs `launch.py up` every
# five minutes, which restarts a replica that died; until it is back the client retries its calls on the others.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
K=${K:-4}; PASSES=${PASSES:-3}; REC=${REC:-24}; CPUW=${CPUW:-24}
STAGES="front diar seg asr align quality tag sounds fill"   # sounds (2026-10-28): its own cluster, cluster_sounds.json
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
P=$MELATI_DATA/podcast
L=$P/logs
mkdir -p "$L" "$P/work"
rm -f "$L/FULL_GPU_DONE"                  # deploy/box_full_publish_loop.sh publishes FINAL once this file exists
TAG=$(date +%m%d%H%M)                    # this driver's logs: run_<slice>_<TAG>_pass<n>.log
$M - "$P/sources.jsonl" "$K" <<'EOF'
import json, sys
src, k = sys.argv[1], int(sys.argv[2])
lines = open(src).read().splitlines()
keep = [l for l in lines if json.loads(l).get("bytes", 1024) >= 1024]     # 50 files of the source are 4 bytes
for i in range(k):
    with open(src.replace(".jsonl", f"_{i}of{k}.jsonl"), "w") as f:
        f.write("".join(l + "\n" for l in keep[i::k]))
print(f"{len(keep)} recordings in {k} slices ({len(lines) - len(keep)} empty files left out)")
EOF
$M serve/launch.py up --config serve/cluster.json
$M serve/launch.py up --config serve/cluster_sounds.json
( while sleep 300; do $M serve/launch.py up --config serve/cluster.json > "$L/watchdog.log" 2>&1
                      $M serve/launch.py up --config serve/cluster_sounds.json >> "$L/watchdog.log" 2>&1; done ) &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT
for pass in $(seq 1 "$PASSES"); do
  pids=()
  for i in $(seq 0 $((K - 1))); do
    $M -m melati.run --sources "$P/sources_${i}of$K.jsonl" --work "$P/work" --stages $STAGES \
        --endpoints $MELATI_DATA/serve/endpoints.json $MELATI_DATA/serve_sounds/endpoints.json \
        --recordings "$REC" --cpu-workers "$CPUW" --io-workers 16 --timing "$P/timing.jsonl" \
        > "$L/run_${i}_${TAG}_pass$pass.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "=== pass $pass"
  grep -h '^\[run\] {' "$L"/run_*_"${TAG}"_pass"$pass".log || true
  left=$(cat "$P"/sources_*of"$K".jsonl | $M -c "import json,os,sys; print(sum(not os.path.exists(os.path.join(sys.argv[1], json.loads(l)['id'], 'fill.jsonl')) for l in sys.stdin))" "$P/work")
  echo "recordings without fill.jsonl: $left"
  [ "$left" -eq 0 ] && break
done
grep -l . "$P"/work/*/error.txt 2>/dev/null | while read -r f; do d=$(dirname "$f"); [ -e "$d/fill.jsonl" ] || echo "$(basename "$d") $(head -c 200 "$f")"; done > "$L/failed.txt"
echo "failed recordings: $(wc -l < "$L/failed.txt") (logs/failed.txt)"
$M serve/launch.py down --config serve/cluster.json
$M serve/launch.py down --config serve/cluster_sounds.json
touch "$L/FULL_GPU_DONE"
echo FULL_GPU_DONE
