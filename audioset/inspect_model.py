"""A zoo model's module tree to two levels, with parameter counts: where a training script can wrap blocks in
gradient checkpointing. RUNS ON THE BOX, in .venv-sed.

    python audioset/inspect_model.py eat-large
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zoo import REPOS  # noqa: E402
from transformers import AutoModel  # noqa: E402

m = AutoModel.from_pretrained(REPOS[sys.argv[1]], trust_remote_code=True)
print(type(m).__name__, "supports_gradient_checkpointing:", getattr(m, "supports_gradient_checkpointing", None))
for n, c in m.named_modules():
    if n.count(".") <= 2:
        k = sum(p.numel() for p in c.parameters())
        if k > 1e6 or n.count(".") <= 1:
            print(f"{n or '(root)':50s} {type(c).__name__:28s} {k / 1e6:8.1f}M")
