#!/bin/bash
# Phase C: the language held AND blank clips trained under the language base detects on them.
#
#   bash deploy/box_asr_sweep4.sh
#
# Measured with the task stated (asr/bench/run_benchmark.py), the language KL fixed the multilingual side
# (25% replay + distill: FLEURS 0.283 with the language detected against base 0.302, identification 0.823
# against 0.837) but the hallucination arms stayed high (music 0.64, other non-speech 0.36): blank clips
# had been trained under a language drawn from 89, so "no words" was learned one language in 89 at a time.
# Every run here uses asr/scripts/blank_langs.py's map (base's detected language per blank clip).
#   GPU 2  all_cv22_r25 + distill, then all + distill
#   GPU 0  all_cv22_r25_tts + distill (the Kannada/Gujarati/Indic arm), after its current label run
#   GPU 4  all_cv22_r25 + label, after its re-scores
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO/asr"
D=$MELATI_DATA/asr
MAP=$D/mixes/blank_langs.json
until [ -s "$MAP" ]; do sleep 60; done
export BLANK_LANG_MAP=$MAP
(bash train/grid.sh 2 all_cv22_r25:64:5e-4:1000:attnmlp:distill all:64:5e-4:1000:attnmlp:distill) > "$D/logs/grid4_2.log" 2>&1 &
(until [ -s "$D/runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000_lidlabel/eval.json" ]; do sleep 60; done
 bash train/grid.sh 0 all_cv22_r25_tts:64:5e-4:1000:attnmlp:distill) > "$D/logs/grid4_0.log" 2>&1 &
(until grep -q REBENCH_DONE $MELATI_LOGS/rebench4b.log 2>/dev/null; do sleep 60; done
 bash train/grid.sh 4 all_cv22_r25:64:5e-4:1000:attnmlp:label) > "$D/logs/grid4_4.log" 2>&1 &
wait
echo ASR_SWEEP4_DONE
