#!/bin/bash
# Re-run stage C (seg) for every recording of a run, and invalidate the later stages only where the
# segments changed, in the shared work dir and in every per-transcriber work_<tag> dir.
#
#   bash deploy/box_reseg.sh <dir>
#
# Written for the same-speaker nested-turn defect (melati/stage_seg.py merge_same_speaker): on the
# podcast pilot 31 of 200 recordings had segments sharing speech or cut short by the padding clamp.
# Stages resume by output existence, so deleting a recording's later outputs is what re-runs them.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
P=${1:?dir with sources.jsonl and work/}
cd "$MELATI_REPO"
export PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
n=0
for d in "$P"/work/*/; do
  [ -e "$d/segments.jsonl" ] || continue
  mv "$d/segments.jsonl" "$d/segments.old.jsonl"
  mv "$d/seg_stats.json" "$d/seg_stats.old.json" 2>/dev/null
  n=$((n + 1))
done
echo "=== re-segmenting $n recordings $(date +%T)"
$M -m melati.run --sources "$P/sources.jsonl" --work "$P/work" --out "$P/export_reseg_unused" \
    --endpoints $MELATI_DATA/serve_eval/endpoints.json --stages seg 2>&1 | grep -E "^\[run\] \{|Traceback" | tail -2
changed=0
for d in "$P"/work/*/; do
  id=$(basename "$d")
  if ! cmp -s "$d/segments.jsonl" "$d/segments.old.jsonl"; then
    changed=$((changed + 1))
    for w in "$P"/work_*/; do
      rm -f "$w/$id"/{asr,align,mos,tags,fill,export}.jsonl "$w/$id/error.txt"
    done
  fi
done
rm -rf "$P/export_reseg_unused"
echo "segments changed in $changed recordings; their later stages will re-run"
$M scripts/audit_segments.py "$P/work"
echo RESEG_DONE
