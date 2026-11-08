#!/bin/bash
# Rebuild the hallucination training corpus and the Common Voice 22 replay pool on the box.
#
#   bash deploy/box_asr_corpus.sh
#
# The original build (the Whisper-Hallucination project's) no longer exists, so the arms are rebuilt with the same scripts and seeds. Sizes follow the
# paper's staged corpus: 24,212 blank-label clips and 8,315 positives. Every arm resumes
# (existing FLAC files are reused), so a re-run after a failure only does what is missing.
#
#   blank   FSD50K dev, voice-free (4,000 music + 14,000 noise)   build_nonspeech_arm.py --split dev
#   blank   FMA shards 2-12, 6,000 excerpts                       build_music_arm.py --shards 2-12
#   blank   synthetic floors, seed 1000                           make_silence_arm.py
#   text    Malaysian-Emilia podcast speech, 4,000 + 4,000        mine_emilia.py
#   text    reduplication, held-out syllables, seed 1000          make_reduplication_arm.py
#   replay  Common Voice 22, <=600 per language (+20 test)        build_cv22.py
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO/asr"
export HF_HOME=$MELATI_DATA/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
A=$D/audio_train
mkdir -p "$A" "$D/logs"

step() {  # name, command...: one log per arm, and a done-file so a re-run skips it
  local name=$1; shift
  if [ -f "$D/logs/$name.done" ]; then echo "[skip] $name"; return 0; fi
  echo "=== $name $(date +%T)"
  if "$@" > "$D/logs/$name.log" 2>&1; then
    touch "$D/logs/$name.done"; tail -2 "$D/logs/$name.log"
  else
    echo "!! $name FAILED (exit $?): tail of $D/logs/$name.log"; tail -15 "$D/logs/$name.log"
  fi
}

# synthetic arms first: seconds, and they prove the venv works
step silence  $PY scripts/make_silence_arm.py --seed 1000 --out "$A/silence"
step redup_cv $PY scripts/make_reduplication_arm.py --seed 1000 --patterns cv \
                 --labels ti ke pu de bo gi lu ne mo gu --out "$A/reduplication_cv"
step redup_laugh $PY scripts/make_reduplication_arm.py --seed 1000 --patterns laugh \
                 --labels o u --out "$A/reduplication_laugh"

# network-bound arms in parallel: none of them shares a source
step nonspeech $PY scripts/build_nonspeech_arm.py --split dev --seed 1000 --music 4000 --noise 14000 \
                 --out-nonspeech "$A/nonspeech" --out-mix "$D/unused_speech_in_noise" &
step music     $PY scripts/build_music_arm.py --shards 2-12 --tracks 6000 --seed 1000 --out "$A/music" &
step emilia    $PY scripts/mine_emilia.py --target-phrase-clips 4000 --general-clips 4000 \
                 --out "$A/emilia" --raw "$D/emilia_raw" &
step cv22      $PY scripts/build_cv22.py --out "$D/cv22" --per-lang 600 --test-per-lang 20 --workers 48 &
wait

# the corpus proper (blank + text arms; CV22 is NOT under audio_train, so it cannot leak in)
step corpus $PY scripts/build_training_corpus.py --audio-root "$A" --out "$D/corpus" \
                 --nonspeech-hours 105 --positive-hours 60
echo ASR_CORPUS_DONE
