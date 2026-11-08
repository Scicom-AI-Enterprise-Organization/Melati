#!/bin/bash
# Distil the released emotion head from two teachers and set both students against the release, on the same held-out
# clips (nEMO, NonverbalTTS's VoxCeleb half):
#   clean   the mean of heads trained only on the release's licence-clean corpora (Whisper, Melati's Whisper fine-tune,
#           w2v-BERT 2.0, WavLM; not the head on emotion2vec+ features, which carries emotion2vec+'s own licence)
#   strong  the seven models the pipeline averages (Melati's five heads on all 21 corpora, MERaLiON-SER, emotion2vec+)
# Students are the release's head on frozen whisper-large-v3 features, trained on the licence-clean corpora (or, for
# the ceiling, all 21) and pulled towards the teacher on 30,000 unlabelled Malay and English podcast clips from the
# published shards (emotion/pool.py). Three seeds for each main arm, two other strengths of the pull for two of them.
#
#   bash deploy/box_emo_distill.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPUS=${GPUS:-"0 1 2 4"}
cd "$MELATI_REPO"
export HF_HOME=$MELATI_DATA/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=$MELATI_REPO
A=$MELATI_ROOT/.venv-asr/bin/python
E=$MELATI_DATA/emo
O=$MELATI_DATA/results/emotion/distill
H=$MELATI_DATA/models/ser_heads_release
D=$MELATI_DATA/models/ser_heads_distill
LOG=$MELATI_LOGS
mkdir -p $O $D

# 1. the pool, and the strong teacher's probabilities for it
[ -s $E/pool/manifest.jsonl ] || $A emotion/pool.py --hf $MELATI_DATA/podcast/hf \
    --work $MELATI_DATA/podcast/work --n 30000 --out $E/pool
wc -l $E/pool/manifest.jsonl

# 2. the pool's features, one encoder a GPU (a corpus whose features exist is skipped)
i=1
for enc in whisper whisper_ft w2vbert wavlm; do
    gpu=$(echo $GPUS | cut -d" " -f$i)
    CUDA_VISIBLE_DEVICES=$gpu $A emotion/extract.py --manifest $E/pool/manifest.jsonl --encoder $enc --out $E/feats \
        > $LOG/emodistill_extract_$enc.log 2>&1 &
    i=$((i + 1))
done
wait
grep -h "podcast_pool" $LOG/emodistill_extract_*.log

G1=$(echo $GPUS | cut -d" " -f1)
export CUDA_VISIBLE_DEVICES=$G1
# 3. clean heads on the other encoders (the Whisper one is the release's), and the clean teacher on the held-out clips
for enc in whisper_ft w2vbert wavlm; do
    [ -s $H/$enc.pt ] || $A emotion/probe.py --manifest $E/manifest_release.jsonl --feats $E/feats --encoder $enc \
        --no-loco --out $O/clean_probe_$enc.json --save-probs $O/clean_probs_$enc.npz --save-model $H/$enc.pt \
        > $LOG/emodistill_probe_$enc.log 2>&1
done
$A emotion/ensemble.py --members $MELATI_DATA/results/emotion/release/probs_whisper.npz $O/clean_probs_whisper_ft.npz \
    $O/clean_probs_w2vbert.npz $O/clean_probs_wavlm.npz --out $O/clean_teacher.json > $O/clean_teacher.log 2>&1

# 4. the students, against the two baselines (the release's recipe on the clean corpora, and on all 21)
CLEAN_HEADS="whisper=$H/whisper.pt whisper_ft=$H/whisper_ft.pt w2vbert=$H/w2vbert.pt wavlm=$H/wavlm.pt"
run() {          # name manifest teacher lam seed
    $A emotion/distill.py --manifest $E/$2 --feats $E/feats --pool $E/pool --teacher $3 --clean-heads $CLEAN_HEADS \
        --lam $4 --seed $5 --out $O/$1_s$5.json --save-probs $O/$1_s$5.npz --save-model $D/$1_s$5.pt \
        2>&1 | grep -v -i warn | tail -1
}
for s in 0 1 2; do
    run base_clean manifest_release.jsonl none 0 $s
    run kd_clean manifest_release.jsonl clean 1 $s
    run kd_strong_cleanlabels manifest_release.jsonl strong 1 $s
    run base_all manifest.jsonl none 0 $s
    run kd_strong_all manifest.jsonl strong 1 $s
done
for l in 0.5 2; do
    run kd_clean_lam$l manifest_release.jsonl clean $l 0
    run kd_strong_all_lam$l manifest.jsonl strong $l 0
done
echo EMO_DISTILL_DONE
