#!/bin/bash
# PodcastFillers (Zhu, Caceres and Salamon, Interspeech 2022; zenodo.org/records/7121457): the filler
# and non-verbal benchmark in the podcast domain.
#
#   bash deploy/box_fetch_podcastfillers.sh
#
# 199 English podcast episodes (145 h, SoundCloud, CC-BY / BY-SA / BY-ND 3.0 audio) with 85,803
# hand-labelled events timed in the episode: Uh 17,907, Um 17,078, Breath 8,288, Laughter 6,623,
# and words, repetitions, music, noise. Episodes are split train / validation / test by SHOW, so
# no test speaker is in train. The annotations are Adobe's, for non-commercial research only:
# they are used here to score Melati and are never redistributed with it.
#
# The archive is a `zip -s` split set (4 parts + .zip, 25.3 GB); 7-Zip reads it as it is.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
D=$MELATI_DATA/bench/podcastfillers
URL=https://zenodo.org/records/7121457/files
mkdir -p "$D/zips"
for f in PodcastFillers.csv PodcastFillers.z01 PodcastFillers.z02 PodcastFillers.z03 PodcastFillers.zip; do
  for attempt in 1 2 3 4 5 6; do
    curl -sS -L -C - --retry 5 --retry-delay 10 -o "$D/zips/$f" "$URL/$f?download=1" && break
    echo "$f: attempt $attempt failed, retrying"; sleep 20
  done
  ls -la "$D/zips/$f"
done
mv -f "$D/zips/PodcastFillers.csv" "$D/PodcastFillers.csv" 2>/dev/null || true
if [ ! -f "$D/.extracted" ]; then
  $MELATI_ROOT/bin/7zz x "$D/zips/PodcastFillers.zip" -o"$D" -y -bso0 -bsp0 && touch "$D/.extracted"
fi
find "$D" -maxdepth 4 -type d | head -30
echo FETCH_PF_DONE
