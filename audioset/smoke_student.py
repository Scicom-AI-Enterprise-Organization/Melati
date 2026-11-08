"""A student's code path end to end on a couple of real evaluation clips, on the CPU: features, forward, the loss and
one backward pass through the checkpointed blocks, so a long run does not find a bug hours in. RUNS ON THE BOX.

    python audioset/smoke_student.py dasheng-1.2b
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load  # noqa: E402
from train import checkpointed, logits_of, student  # noqa: E402

name = sys.argv[1]
m, feat = student(name)
W, _, _, Y = load("eval")
w = torch.from_numpy(np.asarray(W[:2], np.float32))
f = feat(w, train=True)
print("features", tuple(f.shape), f"{f.min().item():.1f}..{f.max().item():.1f}", flush=True)
n = checkpointed(m)
m.train()
t0 = time.time()
out = logits_of(m(f))
loss = F.binary_cross_entropy_with_logits(out.float(), torch.from_numpy(Y[:2].astype(np.float32)))
loss.backward()
grads = [p.grad for p in m.parameters() if p.grad is not None]
print(f"logits {tuple(out.shape)}, loss {loss.item():.4f}, {n} blocks checkpointed, {len(grads)} tensors with "
      f"gradients, head grad norm {m.head.weight.grad.norm().item():.4f}, {time.time() - t0:.0f} s", flush=True)
print("SMOKE_OK", flush=True)
