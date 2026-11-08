"""What EAT's extract_features returns for a 10.24 s input: its shape, and how the patch tokens map onto time and
frequency (a ramp along time and a ramp along frequency, each fed alone, show which token axis follows which), for a
frame-level head. RUNS ON THE BOX, on the CPU.

    python audioset/inspect_tokens.py eat-large
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zoo import REPOS  # noqa: E402
from transformers import AutoModel  # noqa: E402

m = AutoModel.from_pretrained(REPOS[sys.argv[1]], trust_remote_code=True).eval()
with torch.no_grad():
    z = m.extract_features(torch.zeros(1, 1, 1024, 128))
    print("extract_features:", tuple(z.shape))
    p = m.model.local_encoder
    print("patch embed:", type(p).__name__, {k: getattr(p, k) for k in ("img_size", "patch_size", "grid_size", "num_patches")
                                           if hasattr(p, k)})
    t = torch.zeros(1, 1, 1024, 128); t[:, :, :512] = 1.0          # loud in the first half of time only
    f = torch.zeros(1, 1, 1024, 128); f[:, :, :, :64] = 1.0         # loud in the low half of frequency only
    et, ef = p(t), p(f)
    print("patch tokens:", tuple(et.shape))
    nt = et.norm(dim=-1)[0]; nf = ef.norm(dim=-1)[0]
    print("time ramp, token norms by index (first 16):", [round(v, 1) for v in nt[:16].tolist()])
    print("freq ramp, token norms by index (first 16):", [round(v, 1) for v in nf[:16].tolist()])
