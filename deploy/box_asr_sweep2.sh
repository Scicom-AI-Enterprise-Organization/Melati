#!/bin/bash
# The second ASR sweep, with the decoder fed its prefix and full fine-tunes on fp32 master weights
# (asr/train/finetune_whisper.py, top). Phase A answers the first questions in one pass:
#
#   does the fix alone keep FLEURS?            LoRA r64 5e-4 (the paper's best) on `all`
#   does Common Voice replay still help?       the same on all_cv22_r25 and all_cv22_r50
#   LoRA or full parameters?                   full fine-tunes at the paper's 5e-6, 1e-5, 2e-5
#   which LoRA family / rank / rate?           attention-only r64; rank 16 and 128; 1e-4
#
#   bash deploy/box_asr_sweep2.sh
#
# GPUs 0 and 1 still benchmark the first sweep's `all` and
# `all_cv22_r25` runs, kept as the record of the prefix bug; their chains start when those finish.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO/asr"
D=$MELATI_DATA/asr
M=$MELATI_ROOT/.venv-melati/bin/python

echo "=== stop the first sweep's 3000-step runs (trained without the prefix) $(date +%T)"
for pid in $(ps -eo pid,args | grep "[f]inetune_whisper.py.*_s3000" | awk '{print $1}'); do
  kill "$pid" && echo "killed $pid"
done
echo "=== free GPU 2 (the Qwen3-Omni server; brought back for the pipeline later)"
(cd $MELATI_REPO && $M serve/launch.py down --config serve/cluster_omni.json 2>&1 | tail -2)
sleep 20

wait_eval() { until [ -s "$D/runs/$1/eval.json" ]; do sleep 60; done; }

bash train/grid.sh 2 all:64:5e-4 all:0:1e-5                > "$D/logs/grid2_2.log" 2>&1 &
bash train/grid.sh 3 all_cv22_r25:64:5e-4 all:0:5e-6       > "$D/logs/grid2_3.log" 2>&1 &
bash train/grid.sh 4 all_cv22_r50:64:5e-4 all:0:2e-5       > "$D/logs/grid2_4.log" 2>&1 &
(wait_eval v3_lora_r64_all_lr5e-4_s1000;
 bash train/grid.sh 0 all:64:5e-4:1000:attn all:16:5e-4)   > "$D/logs/grid2_0.log" 2>&1 &
(wait_eval v3_lora_r64_all_cv22_r25_lr5e-4_s1000;
 bash train/grid.sh 1 all:64:1e-4 all:128:5e-4)            > "$D/logs/grid2_1.log" 2>&1 &
wait
echo ASR_SWEEP2_DONE
