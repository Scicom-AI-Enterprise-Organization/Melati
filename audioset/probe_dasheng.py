"""Dasheng as a student: load it, run a batch of 10 s clips in bf16 through its own feature extractor, and report the
output's shape, the time a batch takes, and the GPU memory it holds, inference and one training step. RUNS ON THE BOX.

    CUDA_VISIBLE_DEVICES=3 python audioset/probe_dasheng.py mispeech/dasheng-1.2B
"""
import sys
import time

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

repo = sys.argv[1]
fe = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True)
m = AutoModel.from_pretrained(repo, trust_remote_code=True).cuda()
print(type(m).__name__, f"{sum(p.numel() for p in m.parameters()) / 1e9:.2f}B parameters", flush=True)
xs = [np.random.randn(160000).astype(np.float32) * 0.1 for _ in range(8)]
inp = {k: v.cuda() for k, v in fe(xs, sampling_rate=16000, return_tensors="pt").items()}
print({k: tuple(v.shape) for k, v in inp.items()}, flush=True)
m.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(**inp)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        out = m(**inp)
    torch.cuda.synchronize()
print("outputs:", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)} if isinstance(out, dict) else
      [tuple(o.shape) for o in out if torch.is_tensor(o)], flush=True)
print(f"inference: {(time.time() - t0) / 3 * 1000:.0f} ms a batch of 8, {8 * 3 / (time.time() - t0):.0f} clips/s; "
      f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
m.train()
torch.cuda.reset_peak_memory_stats()
opt = torch.optim.AdamW(m.parameters(), lr=1e-6)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(**inp)
    h = out["last_hidden_state"] if isinstance(out, dict) and "last_hidden_state" in out else (
        out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states else out[0])
    loss = h.float().pow(2).mean()
loss.backward(); opt.step()
print(f"one training step (no checkpointing, batch 8): peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB",
      flush=True)
