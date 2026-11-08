#!/bin/bash
# The non-verbal head through the real pipeline, on one pilot recording: the production tag server (seven emotion
# models and nv_whisper), stage G (tag) and export, then what labelled each gap and what export made of it.
#
#   bash deploy/box_nv_smoke.sh
#
# Restarts the tag server (serve/cluster_tag2.json) so it loads the current EMOTION_MODELS, and leaves it up.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
M=$MELATI_ROOT/.venv-melati/bin/python
# a healthy server is not restarted by `up`, and one started before nv_whisper joined EMOTION_MODELS lacks it
$M serve/launch.py down --config serve/cluster_tag2.json
$M serve/launch.py up --config serve/cluster_tag2.json
S=$MELATI_DATA/smoke_nv
REC=${1:-r03c6a9c8d8eb}
rm -rf $S && mkdir -p $S/work/$REC
grep "\"$REC\"" $MELATI_DATA/pilot/sources.jsonl > $S/sources.jsonl
for f in $MELATI_DATA/pilot/work_rel/$REC/*; do
    case "$(basename $f)" in tags.jsonl|export.jsonl) ;; *) ln -s "$f" $S/work/$REC/ ;; esac
done
$M -m melati.run --sources $S/sources.jsonl --work $S/work --out $S/export \
    --endpoints $MELATI_DATA/serve_tag2/endpoints.json --stages tag export
$M - $S/work/$REC <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
d = Path(sys.argv[1])
by, top = Counter(), Counter()
for row in map(json.loads, open(d / "tags.jsonl")):
    for c in row.get("nv_gaps") or []:
        by[c.get("by")] += 1
        top[max(c["dist"], key=c["dist"].get)] += 1
ev = Counter()
for row in map(json.loads, open(d / "export.jsonl")):
    ann = json.loads(row["annotation"]) if isinstance(row.get("annotation"), str) else row.get("annotation") or {}
    for e in ann.get("events") or []:
        ev[(e.get("kind"), e.get("src"))] += 1
print("candidates labelled by:", dict(by))
print("their most probable label:", dict(top.most_common()))
print("exported events (kind, source):", dict(ev))
assert by and set(by) == {"heads"}, by
PY
echo NV_SMOKE_DONE
