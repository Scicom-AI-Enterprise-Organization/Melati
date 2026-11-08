#!/usr/bin/env python3
"""Where a batch's time goes in melati_engine.py's forward: each member's share at one batch size (bf16 weights,
autocast, eager), and the CUDA kernels that take the most time over the whole forward. RUNS ON THE BOX, in .venv-sed.

    python audioset/profile_engine.py --path $MELATI_DATA/release/Melati-AudioSet-Ensemble [--batch 64]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "hub")]
from melati_ensemble import CLIP, Ensemble, features  # noqa: E402


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--cudnn-benchmark", action="store_true", help="let cuDNN time its convolution algorithms")
    ap.add_argument("--sdpa", default=None, choices=["flash", "efficient", "cudnn"], help="force one attention kernel")
    ap.add_argument("--compile", action="store_true", help="torch.compile each member's network")
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    ens = Ensemble(a.path, device="cuda").bf16_weights()
    if a.compile:
        for m in ens.members.values():
            if hasattr(m.net, "encoder") and hasattr(m.net, "patch_embedding"):         # BEATs: its encoder
                m.net.encoder = torch.compile(m.net.encoder)
            else:
                m.net = torch.compile(m.net)
    if a.sdpa:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_kernel([{"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                      "cudnn": SDPBackend.CUDNN_ATTENTION}[a.sdpa]]).__enter__()
    w = (torch.randn(a.batch, CLIP, device="cuda") * 0.1)
    kinds = {m.feature for m in ens.members.values()}
    with torch.no_grad():
        print(f"batch {a.batch}: features {timed(lambda: features(w, kinds)):.1f} ms", flush=True)
        feats = features(w, kinds)
        for k, m in ens.members.items():
            def run(m=m):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(feats)
            print(f"  {k}: {timed(run):.1f} ms", flush=True)
        print(f"  all: {timed(lambda: ens.forward(w)):.1f} ms", flush=True)
        if a.no_profile:
            return
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            ens.forward(w)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.top), flush=True)
    print(f"sdp flash {torch.backends.cuda.flash_sdp_enabled()}, mem-efficient "
          f"{torch.backends.cuda.mem_efficient_sdp_enabled()}, torch {torch.__version__}", flush=True)


if __name__ == "__main__":
    main()
