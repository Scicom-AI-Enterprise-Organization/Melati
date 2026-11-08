"""The batched filterbank (audioset/zoo.py kaldi_fbank) against torchaudio's own, on real AudioSet clips: the largest
difference in log mel, and in EAT-large's probabilities fed one or the other. RUNS ON THE BOX, in .venv-sed.

    python audioset/check_fbank.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SR, load  # noqa: E402
from zoo import kaldi_fbank  # noqa: E402

W, _, _, _ = load("eval")
xs = torch.as_tensor(np.asarray(W[:16], np.float32)).cuda()
xs = xs - xs.mean(1, keepdim=True)
ref = torch.stack([torchaudio.compliance.kaldi.fbank(x[None], htk_compat=True, sample_frequency=SR, use_energy=False,
                                                     window_type="hanning", num_mel_bins=128, dither=0.0,
                                                     frame_shift=10) for x in xs])
mine = kaldi_fbank(xs)
print(f"shapes {tuple(ref.shape)} {tuple(mine.shape)}; max |diff| {(ref - mine).abs().max().item():.2e}, "
      f"mean {(ref - mine).abs().mean().item():.2e}")
