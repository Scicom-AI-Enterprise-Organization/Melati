#!/bin/bash
# AudioSet's frame-level taggers from PretrainedSED (Schmid et al., ICASSP 2025; MIT): BEATs, ATST-F, M2D, ASiT and
# fPaSST fine-tuned on AudioSet's strong (time-stamped) labels, 447 classes every 40 ms. Its own venv, so the tag
# servers' environment (.venv-melati) is not touched; checkpoints land in its resources/ on first use.
#
#   bash deploy/box_sed_setup.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
E=$MELATI_ROOT/ext/PretrainedSED
V=$MELATI_ROOT/.venv-sed
checkout https://github.com/fschmid56/PretrainedSED "$E"
[ -x $V/bin/python ] || uv venv -q $V --python 3.12
# one resolution, torch pinned: a second `uv pip install` let timm's torchvision pull torch 2.14 over torchaudio 2.8;
# setuptools<81 keeps pkg_resources, which dcase_util (under sed_scores_eval) imports; pyarrow<19 still runs on
# the numpy<2 PretrainedSED asks for; transformers 4.46 runs the remote code of EAT and SSLAM (audioset/zoo.py), which
# the 5.x releases break, and so serves Melati's own AudioSet tagger (serve/sounds_server.py, with fastapi and uvicorn
# at the tag servers' versions)
uv pip install -q --python $V/bin/python "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0" "numpy<2" cython \
    "setuptools<81" wheel librosa pandas timm nnAudio "av>=10.0.0" "h5py>=3.8.0" "jsonpickle>=3.0.1" \
    "transformers==4.46.3" "fastapi==0.159.1" "uvicorn==0.54.0" "huggingface_hub[hf_transfer]" \
    intervaltree more-itertools soundfile "pyarrow<19" "datasets<4" soxr pyloudnorm matplotlib "pytorch-lightning>=2.0.0" \
    scikit-learn
# sed_scores_eval's build needs numpy and does not declare it
uv pip install -q --python $V/bin/python --no-build-isolation "sed_scores_eval==0.0.3"
# Melati's AudioSet tagger, which the sounds service loads when it starts (serve/cluster_sounds.json), at its pin
HF_HOME=$MELATI_DATA/hf $V/bin/hf download Scicom-intl/Melati-AudioSet-EAT-Large --revision \
    "$(pin Scicom-intl/Melati-AudioSet-EAT-Large)" 2>&1 | tail -1
cd $E
CUDA_VISIBLE_DEVICES=${GPU:-1} $V/bin/python inference.py --cuda --model_name=BEATs \
    --audio_file=test_files/752547__iscence__milan_metro_coming_in_station.wav | tail -12
ls -la resources
echo SED_SETUP_DONE
