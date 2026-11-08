#!/bin/bash
# The ASR-dependent half of a Melati run (asr, align, quality, tag, fill, export), for ONE ASR model,
# into its own work and export directories, so several models' runs sit side by side on the same
# front/diar/seg outputs (deploy/box_pilot_front.sh), which are linked, not copied.
#
#   bash deploy/box_pilot_rest.sh <dir> <asr model> <tag> [gpu] [port]
#
#   <dir>/work_<tag>/<rec>/   links to the shared outputs + this model's stage files
#   <dir>/export_<tag>/       FLAC + manifest (melati.run --assemble)
#   <dir>/timing_<tag>.jsonl  wall clock per pass
#
# Starts a vLLM server for the model on <gpu> (default 5) beside the eval cluster, whose align,
# mos and tag servers it uses, on <port> (default 18130): two runs at once need two ports, or the
# second reuses the first's server and the first's shutdown kills it mid-run. EXTRA_ENDPOINTS adds
# more endpoint files, e.g. the omni cluster's, which turns on the non-verbal labels of stage G.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
P=${1:?dir}; MODEL=${2:?asr model (hub id or path)}; TAG=${3:?tag}; GPU=${4:-5}; PORT=${5:-18130}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
W=$P/work_$TAG
SHARED="sep.json sed.npz stems.npz speech16k.flac speech24k.flac vocal16k.flac vocal24k.flac diar.json segments.jsonl seg_stats.json"
n=0
for d in "$P"/work/*/; do
  id=$(basename "$d"); [ -e "$d/segments.jsonl" ] || continue     # an empty one is a recording with no speech: zero rows
  mkdir -p "$W/$id"
  for f in $SHARED; do [ -e "$d/$f" ] && ln -sf "$d/$f" "$W/$id/$f"; done
  n=$((n + 1))
done
echo "=== $n recordings linked into $W"
CFG=$P/cluster_asr_$TAG.json
$M - "$MODEL" "$GPU" "$CFG" "$TAG" "$PORT" <<'EOF'
import json, os, sys
model, gpu, out, tag, port = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
base = json.load(open("serve/cluster.json"))
asr = dict(base["services"]["asr"], model=model, port=port, replicas=[[gpu]], gpu_memory_utilization=0.2)
d = os.environ["MELATI_DATA"]
json.dump({"run_dir": f"{d}/serve_asr_{tag}_{port}",
           "endpoints": f"{d}/serve_asr_{tag}_{port}/endpoints.json",
           "startup_timeout_s": 1500, "env": base["env"], "services": {"asr": asr}}, open(out, "w"), indent=1)
EOF
$M serve/launch.py up --config "$CFG"
$M -m melati.run --sources "$P/sources.jsonl" --work "$W" --out "$P/export_$TAG" \
    --endpoints $MELATI_DATA/serve_eval/endpoints.json "$MELATI_DATA/serve_asr_${TAG}_$PORT/endpoints.json" \
                ${EXTRA_ENDPOINTS:-} \
    --stages asr align quality tag fill export --timing "$P/timing_$TAG.jsonl"
$M -m melati.run --assemble --sources "$P/sources.jsonl" --work "$W" --out "$P/export_$TAG"
$M serve/launch.py down --config "$CFG"
echo PILOT_REST_DONE
