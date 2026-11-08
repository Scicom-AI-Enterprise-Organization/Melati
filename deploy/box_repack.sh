#!/bin/bash
# Re-pack every shard of Melati v1 after a change to the export, and publish them in one commit (MSG says what changed).
# The first, 2026-10-27, left out the rows the release audit found wrong (melati.stage_export.release_drop), added
# `squeezed_words`, wrote Mandarin without spaces and moved laughs to 0.97; the second added AudioSet's tags (sniffs,
# throat clearing, whispering, more laughs and coughs; nonverbal/audioset_pass.py must have run first); the third
# (2026-10-28) the `sounds` column and the AudioSet sound tags (the sounds stage must have run over every finished
# recording first: NAME=all HOURS=100000 deploy/box_sounds_pilot.sh). A shard is
# otherwise written once, so the current ones are moved aside first (kept until the repo reads back) and every complete
# block is packed again, including those finished since the last publish. The upload swaps all of them in at once:
# old and new shards have different columns.
#
#   MSG='what changed' bash deploy/box_repack.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REPO=${REPO:-Scicom-intl/Melati}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
P=$MELATI_DATA/podcast
H=$P/hf
L=$P/logs
R=$MELATI_DATA/results/dataset
if [ "${RESUME:-0}" != 1 ]; then                            # RESUME=1: the shards are packed, carry on from there
    OLD=$P/hf_before_repack_${STAMP:-$(date +%m%d%H%M)}     # one folder a re-pack, kept until the repo reads back
    mkdir -p "$OLD"
    mv "$H/data" "$OLD/data"
    mv "$H/left_out" "$OLD/left_out" 2>/dev/null || true
    cp "$H/stats.json" "$H/README.md" "$OLD/"
fi
mkdir -p "$H/data"
$M scripts/build_hf_dataset.py --sources "$P/sources.jsonl" --work "$P/work" --out "$H" --workers "${WORKERS:-32}" \
    > "$L/hf_repack.log" 2>&1
grep '^\[hf\]' "$L/hf_repack.log" | tail -2
$M scripts/check_hf_dataset.py --data "$H/data" --rows 1000 > "$L/hf_check_local.log" 2>&1 || true
tail -2 "$L/hf_check_local.log"; grep -q CHECK_OK "$L/hf_check_local.log"
$M scripts/dataset_card.py --hf "$H" --repo "$REPO"
cp "$H/stats.json" "$R/stats.json"; cp "$H/export_config.json" "$R/export_config.json"
$M scripts/hf_dataset_upload.py --folder "$H" --repo "$REPO" --one-commit \
    --message "${MSG:-Every shard re-packed}" \
    > "$L/hf_upload.log" 2>&1
tail -2 "$L/hf_upload.log"
$M scripts/check_hf_dataset.py --repo "$REPO" --rows 300 > "$L/hf_check_repo.log" 2>&1 || true
tail -2 "$L/hf_check_repo.log"; grep -q CHECK_OK "$L/hf_check_repo.log"
echo REPACK_DONE
