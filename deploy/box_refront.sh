#!/bin/bash
# Re-run stage A (separation, event detection, stems) for every recording of a run whose front outputs predate a
# file the later stages now read (VoxConverse's lacked vocal24k.flac, the 24 kHz speech + effects version the
# quality stage scores). Diarization is kept; segmentation is redone after (deploy/box_reseg.sh), because the VAD
# reads the new speech stem.
#
#   bash deploy/box_refront.sh <dir>
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
P=${1:?dir with sources.jsonl and work/}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
for d in "$P"/work/*/; do
  mkdir -p "$d/old_front"
  for f in sep.json sed.npz stems.npz speech16k.flac speech24k.flac vocal16k.flac vocal24k.flac; do
    [ -e "$d/$f" ] && mv "$d/$f" "$d/old_front/$f"
  done
done
$M -m melati.run --sources "$P/sources.jsonl" --work "$P/work" --out "$P/export_refront_unused" \
    --endpoints $MELATI_DATA/serve_eval/endpoints.json --stages front 2>&1 | grep -E "^\[run\] \{|Traceback" | tail -2
rm -rf "$P/export_refront_unused"
echo REFRONT_DONE
