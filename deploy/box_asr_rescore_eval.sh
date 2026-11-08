#!/bin/bash
# Re-score every run's eval.json from the transcripts already on disk (no GPU): for a change to
# asr/bench/score_eval_run.py, such as the corpus-level WER. Then the summaries and FLEURS report.
#
#   bash deploy/box_asr_rescore_eval.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MELATI_REPO/asr"
PY=$MELATI_ROOT/.venv-asr/bin/python
D=$MELATI_DATA/asr
R=$MELATI_DATA/results/asr
for run in $D/runs/base_large_v3 $D/runs/v3_lora_r64_all_lr5e-4_s1000 $D/runs2/v3_* $D/merges/*; do
  [ -d "$run/bench" ] || continue
  $PY bench/score_eval_run.py --run "$run" --lexicon "$D/lexicon/combined_lexicon.csv" > /dev/null 2>&1 \
    && echo "scored $(basename "$run")" || echo "FAILED $(basename "$run")"
done
$PY summarise.py --runs $D/runs2 --out $R/summary.json > /dev/null
$PY summarise.py --runs $D/runs --out $R/summary_runs1.json > /dev/null
[ -d $D/merges ] && $PY summarise.py --runs $D/merges --out $R/summary_merges.json > /dev/null
$PY lid_report.py --roots $D/runs $D/runs2 $( [ -d $D/merges ] && echo $D/merges ) --out $R/lid.json > /dev/null
$PY scripts/served_failures.py --runs $D/runs/base_large_v3 $D/runs/v3_lora_r64_all_lr5e-4_s1000 $D/runs2/v3_* \
    $( [ -d $D/merges ] && ls -d $D/merges/* ) --out $R/served_failures.json > /dev/null
echo RESCORE_EVAL_DONE
