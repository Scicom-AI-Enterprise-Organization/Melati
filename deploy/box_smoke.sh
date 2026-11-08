#!/bin/bash
# One recording through every stage, to check a setup: a VoxConverse test recording (CC BY 4.0; one file of its test
# split, 367 MB, fetched once), serve/cluster.json and serve/cluster_sounds.json cut to one replica a service, the
# pipeline and the export's rows summed up, then both clusters down again whatever happened. The services keep the
# configs' memory settings, which were set for GPUs of 143 GB: edit them for smaller ones.
#
#   bash deploy/box_smoke.sh [GPU for asr, diar, align] [GPU for front, mos, tag, sounds]       (default: 0 for both)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
G1=${1:-0}
G2=${2:-$G1}
V=$MELATI_ROOT/.venv-melati/bin
D=$MELATI_DATA/smoke
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1
mkdir -p "$D"
[ -s "$D/vox/data/test-00006-of-00011.parquet" ] || "$V/hf" download diarizers-community/voxconverse \
    data/test-00006-of-00011.parquet --repo-type dataset --revision "$(pin diarizers-community/voxconverse)" \
    --local-dir "$D/vox" 2>&1 | tail -1
rm -rf "$D/audio" "$D/work" "$D/export"
"$V/python" scripts/dev_voxconverse.py --parquet-dir "$D/vox/data" --split test --n 1 --out "$D/audio"
"$V/python" -m melati.sources --dir "$D/audio" --list "$D/sources.jsonl"
"$V/python" - "$D" "$G1" "$G2" <<'PY'
import json, sys
from pathlib import Path
d, g1, g2 = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
gpu = {"asr": g1, "diar": g1, "align": g1, "front": g2, "mos": g2, "tag": g2, "sounds": g2}
for name in ("cluster", "cluster_sounds"):
    c = json.loads(Path(f"serve/{name}.json").read_text())
    for k, s in c["services"].items():
        s["replicas"] = [[gpu[k]]]
    (d / f"{name}.json").write_text(json.dumps(c, indent=1))
print("one replica a service:", gpu)
PY
down() {
    "$V/python" serve/launch.py down --config "$D/cluster.json" || true
    "$V/python" serve/launch.py down --config "$D/cluster_sounds.json" || true
}
trap down EXIT
"$V/python" serve/launch.py up --config "$D/cluster.json"
"$V/python" serve/launch.py up --config "$D/cluster_sounds.json"
"$V/python" -m melati.run --sources "$D/sources.jsonl" --work "$D/work" --out "$D/export" --recordings 1 \
    --endpoints "$MELATI_DATA/serve/endpoints.json" "$MELATI_DATA/serve_sounds/endpoints.json"
"$V/python" -m melati.run --sources "$D/sources.jsonl" --work "$D/work" --out "$D/export" --assemble
"$V/python" - "$D/export/manifest" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq
rows = [r for f in sorted(Path(sys.argv[1]).glob("*.parquet")) for r in pq.read_table(f).to_pylist()]
assert rows, "the export holds no rows"
mins, tagged, sounds = sum(r["duration"] for r in rows) / 60, sum("[" in r["tagged_text"] for r in rows), \
    sum(bool(r["sounds"]) for r in rows)
print(f"{len(rows)} rows, {mins:.1f} min, {len(rows[0])} columns; {tagged} with point tags, {sounds} with sounds")
for r in rows[:3]:
    print(f"  {r['id']} {r['duration']:.1f} s: {r['tagged_text'][:120]}")
PY
echo SMOKE_DONE
