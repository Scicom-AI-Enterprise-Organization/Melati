#!/bin/bash
# Phase B: the fixed trainer with the language token kept (finetune_whisper.py --lid-loss).
#
#   bash deploy/box_asr_sweep3.sh
#
# The fixed sweep (runs2) fed the decoder its prefix and the hallucination arms fell further than
# the paper's (fixed 50% replay: non-speech 0.90 -> 0.04), but FLEURS failures went 172 -> 468-707:
# the model obeys a language token it now guesses wrong (romanised Arabic, Yoruba as English).
# This phase holds that token: `distill` keeps base's identification (KL), `label` supervises it on
# speech clips; LoRA r64 5e-4 on `all` and on `all_cv22_r25`, and a full fine-tune with distill.
#
# GPUs 0, 1 and 2 still had phase-A jobs queued (ranks 16 and 128, a full fine-tune at 1e-5, all
# with the token untrained): their grid loops are stopped, the benchmark each GPU is running
# finishes, and the jobs below start when its eval.json exists. GPUs 3 and 4 finish the two full
# fine-tunes they are training (5e-6, 2e-5), which still answer LoRA against full parameters.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO/asr"
D=$MELATI_DATA/asr
for g in 0 1 2; do
  for pid in $(ps -eo pid,args | grep "[t]rain/grid.sh $g " | awk '{print $1}'); do
    kill "$pid" && echo "stopped the phase-A loop of gpu $g (pid $pid); its running benchmark continues"
  done
done
wait_eval() { until [ -s "$D/runs2/$1/eval.json" ]; do sleep 60; done; }
(wait_eval v3_lora_r64_all_lr5e-4_s1000
 bash train/grid.sh 2 all:64:5e-4:1000:attnmlp:distill all_cv22_r25:64:5e-4:1000:attnmlp:distill) > "$D/logs/grid3_2.log" 2>&1 &
(wait_eval v3_loraattn_r64_all_lr5e-4_s1000
 bash train/grid.sh 0 all:64:5e-4:1000:attnmlp:label all_cv22_r25:64:5e-4:1000:attnmlp:label) > "$D/logs/grid3_0.log" 2>&1 &
(wait_eval v3_lora_r64_all_lr1e-4_s1000
 bash train/grid.sh 1 all:0:1e-5:1000:attnmlp:distill all_cv22_r25:0:1e-5:1000:attnmlp:distill) > "$D/logs/grid3_1.log" 2>&1 &
wait
echo ASR_SWEEP3_DONE
