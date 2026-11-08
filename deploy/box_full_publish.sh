#!/bin/bash
# Publish Melati v1 to Hugging Face from what deploy/box_full.sh has finished so far: pack the blocks of 64
# recordings that are complete into shards (scripts/build_hf_dataset.py; a shard is written once), check them, write
# the card, upload what is new, and check the repo reads back before it is made public. Run it again as the run
# finishes more, and once with FINAL=1 after it, which packs every block and leaves out what failed for good.
#
#   bash deploy/box_full_publish.sh
#   FINAL=1 bash deploy/box_full_publish.sh
#
# The numbers the card and the paper quote are kept in $MELATI_DATA/results/dataset.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REPO=${REPO:-Scicom-intl/Melati}
FINAL_FLAG=""; [ "${FINAL:-0}" = 1 ] && FINAL_FLAG="--final"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=$MELATI_REPO
M=$MELATI_ROOT/.venv-melati/bin/python
P=$MELATI_DATA/podcast
H=$P/hf
L=$P/logs
R=$MELATI_DATA/results/dataset
mkdir -p "$L" "$R"
# the vocal sounds' AudioSet tags (laugh, cough, sniff, throat clearing, whisper; nonverbal/audioset_pass.py) over
# what has finished since the last publish: not a stage of the run, and build_hf_dataset.py packs no block before it
ASPASS_LOGS=$L bash deploy/box_audioset_pass.sh ${ASPASS_GPUS:-0 1 2 4 5} > "$L/aspass.log" 2>&1
tail -2 "$L/aspass.log"
$M scripts/build_hf_dataset.py --sources "$P/sources.jsonl" --work "$P/work" --out "$H" --workers "${WORKERS:-16}" \
    $FINAL_FLAG > "$L/hf_build.log" 2>&1
grep '^\[hf\]' "$L/hf_build.log" | tail -2
$M scripts/check_hf_dataset.py --data "$H/data" --rows 1000 > "$L/hf_check_local.log" 2>&1 || true
tail -2 "$L/hf_check_local.log"; grep -q CHECK_OK "$L/hf_check_local.log"
$M scripts/dataset_card.py --hf "$H" --repo "$REPO"
cp "$H/stats.json" "$R/stats.json"; cp "$H/export_config.json" "$R/export_config.json"
$M scripts/hf_dataset_upload.py --folder "$H" --repo "$REPO" > "$L/hf_upload.log" 2>&1
tail -2 "$L/hf_upload.log"
$M scripts/check_hf_dataset.py --repo "$REPO" --rows 300 > "$L/hf_check_repo.log" 2>&1 || true
tail -2 "$L/hf_check_repo.log"; grep -q CHECK_OK "$L/hf_check_repo.log"
$M scripts/hf_dataset_upload.py --repo "$REPO" --public
echo FULL_PUBLISH_DONE
