#!/bin/bash
# Fetch Melati v1's source audio and the two earlier Emilia outputs of the same recordings.
#
#   bash deploy/box_fetch_podcast.sh
#
# source  malaysia-ai/malaysian-podcast-youtube   12-part split zip, 125.7 GB, 19,092 files, 2,233.8 h
# before  mesolitica/Malaysian-Emilia             v1: upstream Emilia-Pipe, incl. its 32 kbit/s MP3 export
# before  mesolitica/Malaysian-Emilia-v2          v2: the mesolitica/Emilia fork that fixed Amphion #436
#
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
PY=$MELATI_ROOT/.venv-asr/bin/python
HF=$MELATI_ROOT/.venv-asr/bin/hf
P=$MELATI_DATA/podcast
mkdir -p "$P/zips" "$P/emilia_v1" "$P/emilia_v2"

echo "=== source zips $(date +%T)"
# hf_transfer died once mid-set ("An error occurred while downloading using hf_transfer") and the script
# went on to extract 7 of 12 parts: the plain downloader resumes partial files, and it is retried until
# every part is there. Extraction refuses an incomplete set.
PARTS=(malaysian-podcast.z0{1..9} malaysian-podcast.z1{0,1} malaysian-podcast.zip)
for attempt in 1 2 3 4 5 6; do
  HF_HUB_ENABLE_HF_TRANSFER=0 $HF download malaysia-ai/malaysian-podcast-youtube --repo-type dataset \
      --include 'malaysian-podcast.z*' --local-dir "$P/zips" --max-workers 4 2>&1 | tail -1
  missing=0; for f in "${PARTS[@]}"; do [ -s "$P/zips/$f" ] || missing=$((missing + 1)); done
  [ "$missing" = 0 ] && break
  echo "attempt $attempt: $missing parts missing, retrying"; sleep 30
done
[ "$missing" = 0 ] || { echo "SOURCE INCOMPLETE: $missing parts missing"; exit 1; }

echo "=== Emilia v1 / v2 outputs of the same podcasts $(date +%T)"
$HF download mesolitica/Malaysian-Emilia --repo-type dataset --include 'malaysian-podcast-processed.zip' \
    --local-dir "$P/emilia_v1" 2>&1 | tail -1
$HF download mesolitica/Malaysian-Emilia-v2 --repo-type dataset \
    --include 'malaysian-podcast_processed-*.zip' 'data/malaysian_podcast-*.parquet' \
    --local-dir "$P/emilia_v2" 2>&1 | tail -1

echo "=== extract the split set with 7-Zip $(date +%T)"
# A `zip -s` split archive addresses (disk, offset) pairs: concatenating the parts does NOT give a
# zip Python can read ("zipfiles that span multiple disks are not supported"). 7-Zip reads the set
# as it is, which is also what the dataset's own README does.
if [ ! -f "$P/.extracted" ]; then
  $MELATI_ROOT/bin/7zz x "$P/zips/malaysian-podcast.zip" -o"$P/raw" -y -bso0 -bsp0 && touch "$P/.extracted"
fi
$PY -m melati.sources --dir "$P/raw" --list "$P/sources.jsonl"
echo FETCH_PODCAST_DONE
