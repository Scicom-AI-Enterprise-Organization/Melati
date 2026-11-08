#!/usr/bin/env python3
"""The model card's two snippets, run as the public would: no token, a fresh cache. RUNS ON THE BOX.
The second runs whichever helper the repo ships (melati_nonverbal.py or melati_emotion.py).

    env -u HF_TOKEN python scripts/hf_public_smoke.py Scicom-intl/Melati-NonVerbal-Whisper-Large-V3
    env -u HF_TOKEN python scripts/hf_public_smoke.py Scicom-intl/Melati-Emotion-Whisper-Large-V3
"""
import importlib.util, sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoFeatureExtractor, AutoModel

repo = sys.argv[1]
sr = 16000
rng = np.random.default_rng(0)
clip = rng.normal(0, 0.01, sr).astype(np.float32)

# 1. AutoModel, as the card shows
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.float16).cuda().eval()
fe = AutoFeatureExtractor.from_pretrained(repo)
f = fe([clip], sampling_rate=sr, return_tensors="pt").input_features.cuda()
frames = torch.tensor([int(len(clip) / sr * 50)])
with torch.inference_mode():
    p = model(f, num_frames=frames).logits.softmax(-1)
print("AutoModel:", type(model).__name__, "| encoder", model.encoder.conv1.weight.dtype, "| head",
      model.head_out.weight.dtype, "| probs sum", round(float(p.sum()), 4), "| labels", len(model.config.labels))

# 2. the helper, as the card shows
helper = next(f for f in list_repo_files(repo) if f in ("melati_nonverbal.py", "melati_emotion.py"))
spec = importlib.util.spec_from_file_location(helper[:-3], hf_hub_download(repo, helper))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
if helper == "melati_emotion.py":
    em = m.Emotion.from_pretrained(repo)
    print("label():", em.label(em.classify([clip])[0]), "| thresholds", em.thresholds, "| priors", em.priors)
    print("PUBLIC_SMOKE_OK")
    sys.exit(0)
nv = m.NonVerbal.from_pretrained(repo)
x = rng.normal(0, 10 ** (-50 / 20), 3 * sr).astype(np.float32)
words = [{"s": 0.3, "e": 0.8}, {"s": 1.6, "e": 2.1}]
for w in words:
    a, b = int(w["s"] * sr), int(w["e"] * sr)
    x[a:b] += (0.1 * np.sin(2 * np.pi * 220 * np.arange(b - a) / sr)).astype(np.float32)
a, b = int(1.0 * sr), int(1.3 * sr)
x[a:b] = rng.normal(0, 10 ** (-46 / 20), b - a).astype(np.float32)      # something in the gap
print("tag():", nv.tag(x, words), "| classify:", nv.classify([x[a:b]]).shape, "| thresholds", nv.thresholds)
if nv.frame_thresholds:
    print("frames():", nv.frames(x).shape, nv.frame_classes, "| frame thresholds", nv.frame_thresholds)
print("PUBLIC_SMOKE_OK")
