#!/usr/bin/env python3
"""The Dasheng release against the student it is built from, before any upload: audioset/release.py's build() writes
the folder, its class is imported the way the Hub's dynamic modules import it (the folder as a package) and loaded with
from_pretrained, and on a few evaluation clips its logits must equal train.DashengTagger's, and the feature extractor
the card tells users to call must give what audioset/train.py trained on. RUNS ON THE BOX, in .venv-sed, on the CPU.

    python audioset/test_hub_dasheng.py $MELATI_DATA/models/audioset/dasheng_big/best.pt
"""
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load  # noqa: E402
from release import build  # noqa: E402
from train import dasheng_features, student  # noqa: E402

ckpt = Path(sys.argv[1])
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
W, _, _, _ = load("eval")
w = torch.from_numpy(np.asarray(W[:4], np.float32))
with tempfile.TemporaryDirectory(dir="/dev/shm") as d:              # 4.5 GB of weights: not on the full root disk
    pkg = Path(d) / "melati_hub_test"
    build(ckpt, pkg)
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, d)
    hub = importlib.import_module("melati_hub_test.modeling_melati_dasheng")
    fe = importlib.import_module("melati_hub_test.feature_extraction_dasheng").DashengFeatureExtractor.from_pretrained(pkg)
    m = hub.DashengAudioSetTagger.from_pretrained(pkg).eval()
    print(type(m).__name__, f"{sum(p.numel() for p in m.parameters()) / 1e9:.2f}B parameters, "
          f"{m.config.num_labels} labels, label 0 = {m.config.id2label[0]!r}", flush=True)
    ref, _ = student(ck["init"], ck["state"])
    ref.eval()
    with torch.no_grad():
        f_train = dasheng_features(w)
        f_card = fe([x.numpy() for x in w], sampling_rate=16000)["input_values"]
        a, b = ref(f_train), m(f_train).logits
        c = m(f_card).logits
    print(f"features: train vs the extractor, max |diff| {(f_train - f_card).abs().max().item():.2e} dB", flush=True)
    print(f"logits: student vs release on the same features, max |diff| {(a - b).abs().max().item():.2e}", flush=True)
    print(f"probabilities: student vs release through the extractor, max |diff| "
          f"{(torch.sigmoid(a) - torch.sigmoid(c)).abs().max().item():.2e}", flush=True)
    assert (a - b).abs().max().item() < 1e-4
    assert (torch.sigmoid(a) - torch.sigmoid(c)).abs().max().item() < 1e-3
print("HUB_DASHENG_OK", flush=True)
