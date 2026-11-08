#!/bin/bash
# BEATs fine-tuned on AudioSet-2M (Chen et al., ICML 2023; MIT), a candidate teacher pre-trained unlike CED, EAT and
# SSLAM: its model code from microsoft/unilm into ext/beats, and the iter3+ (AS2M) checkpoints from Microsoft's own
# OneDrive links (the README's), through OneDrive's shares API. Where that refuses, from two Hugging Face mirrors
# that must agree byte for byte. The checkpoints are loaded weights_only (audioset/zoo.py).
#
#   bash deploy/box_beats_fetch.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
E=$MELATI_ROOT/ext/beats
D=$MELATI_DATA/models/beats
mkdir -p "$E" "$D"
for f in BEATs.py backbone.py modules.py quantizer.py Tokenizers.py; do
    [ -s "$E/$f" ] || curl -fsSL "https://raw.githubusercontent.com/microsoft/unilm/master/beats/$f" -o "$E/$f"
done
onedrive() {                                   # $1 file, $2 the README's sharing link
    [ -s "$D/$1" ] && return 0
    local u
    u=$(printf %s "$2" | base64 -w0 | tr '/+' '_-' | tr -d '=')
    curl -fsSL --retry 3 "https://api.onedrive.com/v1.0/shares/u!$u/root/content" -o "$D/$1.part" \
        && mv "$D/$1.part" "$D/$1" || { rm -f "$D/$1.part"; echo "onedrive refused $1"; return 1; }
}
mirrors() {                                    # $1 file: two mirrors, kept only when they agree
    local a=https://huggingface.co/AudioLLMs/SALMONN_7B/resolve/main/beats_path/$1
    local b=https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/$1
    curl -fsSL --retry 3 "$a" -o "$D/$1.a" && curl -fsSL --retry 3 "$b" -o "$D/$1.b"
    if [ "$(sha256sum < "$D/$1.a")" = "$(sha256sum < "$D/$1.b")" ]; then mv "$D/$1.a" "$D/$1"; rm -f "$D/$1.b"
    else echo "mirrors disagree on $1"; rm -f "$D/$1.a" "$D/$1.b"; return 1; fi
}
F=BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
onedrive $F "https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea" || mirrors $F
ls -la "$D"
sha256sum "$D"/*.pt
echo BEATS_FETCH_DONE
