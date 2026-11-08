#!/usr/bin/env python3
"""A Melati AudioSet tagger, distilled: EAT-large already fine-tuned on AudioSet-2M (Chen et al., IJCAI 2024; MIT)
trained further to the average of the teachers' probabilities (audioset/pool.py: CED-base, EAT-large and SSLAM) on the
pool's clips, with mixup and SpecAugment, an exponential moving average of its weights, and its mAP on the evaluation
set read as the teachers' was (audioset/evaluate.py), every --eval-every steps; the best average is kept. RUNS ON THE
BOX, in .venv-sed, one process a GPU:

    CUDA_VISIBLE_DEVICES=0,1,2,4,5 torchrun --nproc_per_node 5 audioset/train.py --pool clean \\
        --out $MELATI_DATA/models/audioset/clean1
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, load  # noqa: E402
from evaluate import mean_ap  # noqa: E402
from zoo import REPOS, kaldi_fbank  # noqa: E402

FRAMES = 1024


def features(w, train=False, tmask=80, fmask=24, n_masks=2):
    """16 kHz rows -> EAT's input [B, 1, 1024, 128], with SpecAugment's time and frequency masks when training."""
    f = kaldi_fbank(w - w.mean(1, keepdim=True))
    f = F.pad(f, (0, 0, 0, max(0, FRAMES - f.shape[1])))[:, :FRAMES]
    f = (f + 4.268) / (4.569 * 2)
    if train:
        B, T, Fq = f.shape
        for _ in range(n_masks):
            t0 = torch.randint(0, T, (B, 1), device=f.device)
            tl = torch.randint(0, tmask + 1, (B, 1), device=f.device)
            ar = torch.arange(T, device=f.device)[None]
            f = f.masked_fill(((ar >= t0) & (ar < t0 + tl))[:, :, None], 0.0)
            f0 = torch.randint(0, Fq, (B, 1), device=f.device)
            fl = torch.randint(0, fmask + 1, (B, 1), device=f.device)
            af = torch.arange(Fq, device=f.device)[None]
            f = f.masked_fill(((af >= f0) & (af < f0 + fl))[:, None, :], 0.0)
    return f[:, None]


_MEL = {}


def dasheng_features(w, train=False, tmask=100, fmask=12, n_masks=2):
    """16 kHz rows -> Dasheng's input [B, 64, frames]: its own feature extractor's mel spectrogram and dB (64 bins,
    512-point FFT, 10 ms hop, up to 8 kHz, 120 dB floor), on the GPU for a batch at once, with SpecAugment's masks
    when training."""
    import torchaudio
    if w.device not in _MEL:
        from transformers import AutoFeatureExtractor
        fe = AutoFeatureExtractor.from_pretrained(REPOS["dasheng-1.2b"], trust_remote_code=True)
        _MEL[w.device] = (torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=fe.n_fft, win_length=fe.win_size, hop_length=fe.hop_size, f_min=fe.f_min,
            f_max=fe.f_max, n_mels=fe.feature_size, center=fe.center).to(w.device),
            torchaudio.transforms.AmplitudeToDB(top_db=120).to(w.device))
    mel, db = _MEL[w.device]
    with torch.autocast("cuda", enabled=False):
        f = db(mel(w.float()))
    if train:
        B, Fq, T = f.shape
        for _ in range(n_masks):
            t0 = torch.randint(0, T, (B, 1), device=f.device)
            tl = torch.randint(0, tmask + 1, (B, 1), device=f.device)
            ar = torch.arange(T, device=f.device)[None]
            f = f.masked_fill(((ar >= t0) & (ar < t0 + tl))[:, None, :], f.min())
            f0 = torch.randint(0, Fq, (B, 1), device=f.device)
            fl = torch.randint(0, fmask + 1, (B, 1), device=f.device)
            af = torch.arange(Fq, device=f.device)[None]
            f = f.masked_fill(((af >= f0) & (af < f0 + fl))[:, :, None], f.min())
    return f


class DashengTagger(torch.nn.Module):
    """Dasheng's encoder (self-supervised on 272k hours; no AudioSet head of its own) with one for AudioSet's 527
    classes on its tokens averaged over time. Its input batch norm keeps its pretrained statistics."""

    def __init__(self, dasheng, n_classes=527):
        super().__init__()
        self.dasheng = dasheng
        d = dasheng.config.encoder_kwargs["embed_dim"]
        self.norm = torch.nn.LayerNorm(d)
        self.head = torch.nn.Linear(d, n_classes)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.constant_(self.head.bias, -5.0)           # AudioSet's classes are rare: start near p = 0.007

    def train(self, mode=True):
        super().train(mode)
        bn = getattr(self.dasheng.encoder, "init_bn", None)
        if bn is not None:
            bn.eval()
        return self

    def forward(self, mel):
        h = self.dasheng(mel)["hidden_states"]
        return self.head(self.norm(h.mean(1)))


def student(name, init_state=None):
    """-> (model, its features function, the zoo name it rebuilds from)."""
    from transformers import AutoModel
    if name == "dasheng-1.2b":
        m = DashengTagger(AutoModel.from_pretrained(REPOS[name], trust_remote_code=True))
        feat = dasheng_features
    else:
        m, feat = AutoModel.from_pretrained(REPOS[name], trust_remote_code=True), features
    if init_state is not None:
        m.load_state_dict(init_state)
    return m, feat


class Pool(torch.utils.data.Dataset):
    """A pool's clips and targets: one array, or for an index-only pool (audioset/pool.py --index-only) rows of the
    shard caches it points into, each opened as a memory map in the worker that first reads it."""

    def __init__(self, name, hard):
        self.W, self.maps = None, {}
        self.name = name
        m = np.load(ROOT / "cache" / f"pool_{name}_meta.npz")
        t = m["teacher"].astype(np.float32)
        self.y = ((1 - hard) * t + hard * m["labels"].astype(np.float32)) if hard else t
        self.files = m["files"].tolist() if "files" in m.files else None
        self.where = m["where"] if "where" in m.files else None

    def __len__(self):
        return len(self.y)

    def row(self, i):
        if self.files is None:
            if self.W is None:                               # opened in the worker, not pickled across
                self.W = np.load(ROOT / "cache" / f"pool_{self.name}_wav16k.npy", mmap_mode="r")
            return self.W[i]
        f, r = self.where[i]
        if f not in self.maps:
            self.maps[f] = np.load(self.files[f], mmap_mode="r")
        return self.maps[f][r]

    def __getitem__(self, i):
        return torch.from_numpy(self.row(i).astype(np.float32)), torch.from_numpy(self.y[i])


def checkpointed(model):
    """Recompute each transformer block in the backward pass rather than keep its activations: EAT-large at a useful
    batch does not fit beside the model servers otherwise. Wraps the blocks of this instance only."""
    from torch.utils.checkpoint import checkpoint
    blocks = getattr(getattr(model, "model", model), "blocks", None)
    if blocks is None and hasattr(model, "dasheng"):
        blocks = model.dasheng.encoder.blocks
    for b in blocks or ():
        b.forward = (lambda f: (lambda *x, **k: checkpoint(f, *x, use_reentrant=False, **k)))(b.forward)
    return len(blocks or ())


def logits_of(out):
    return out if torch.is_tensor(out) else (out.logits if hasattr(out, "logits") else out[0])


@torch.no_grad()
def evaluate(model, W, Y, rank, world, batch=64, feat=None):
    """mAP on the evaluation set, each process scoring every world-th clip and rank 0 gathering them."""
    model.eval()
    idx = np.arange(rank, len(W), world)
    P = []
    for i in range(0, len(idx), batch):
        w = torch.from_numpy(np.asarray(W[idx[i:i + batch]], np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(torch.sigmoid(logits_of(model((feat or features)(w))).float()).cpu().numpy().astype(np.float16))
    P = np.concatenate(P) if P else np.zeros((0, Y.shape[1]), np.float16)
    gathered = [None] * world
    dist.all_gather_object(gathered, (idx, P))
    model.train()
    if rank:
        return None
    full = np.zeros((len(W), Y.shape[1]), np.float32)
    for ix, p in gathered:
        full[ix] = p
    return mean_ap(full, Y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--init", default="eat-large", help="a model in audioset/zoo.py REPOS, or melati:<checkpoint>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=24, help="clips a GPU a step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--mixup", type=float, default=0.5, help="share of batches mixed")
    ap.add_argument("--alpha", type=float, default=10.0, help="mixup's Beta(alpha, alpha)")
    ap.add_argument("--hard", type=float, default=0.0, help="weight of AudioSet's own labels in the target")
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=int, default=1, help="1: gradient checkpointing over the transformer blocks")
    ap.add_argument("--head-lr-mult", type=float, default=20.0, help="a new head's learning rate over the encoder's")
    ap.add_argument("--zero", type=int, default=0, help="1: shard the optimizer state across the GPUs (big students)")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.init.startswith("melati:"):                         # a student this script saved: its architecture, its weights
        ck = torch.load(a.init.split(":", 1)[1], map_location="cpu", weights_only=False)
        model, feat = student(ck["init"], ck["state"])
        base = ck["init"]
    else:
        model, feat = student(a.init)
        base = a.init
    model = model.cuda().train()
    ema = copy.deepcopy(model).eval()           # DDP keeps every process's weights equal, so each keeps the same average
    for p in ema.parameters():
        p.requires_grad_(False)
    n_ckpt = checkpointed(model) if a.checkpoint else 0  # after the copy: the copy's blocks must stay its own
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    ds = Pool(a.pool, a.hard)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=a.seed, drop_last=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers, pin_memory=True,
                                     drop_last=True, persistent_workers=True)
    steps = int(a.epochs * len(dl))
    new = lambda n: n.startswith(("head.", "norm."))  # noqa: E731  a head the init does not have learns faster
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and (p.ndim > 1) == d and new(n) == h],
               "weight_decay": a.wd if d else 0.0, "lr": a.lr * (a.head_lr_mult if h else 1.0)}
              for d in (True, False) for h in (True, False)]
    groups = [g for g in groups if g["params"]]
    if a.zero:                                             # the optimizer's state split across the GPUs
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(groups[0]["params"], optimizer_class=torch.optim.AdamW, lr=groups[0]["lr"],
                                      weight_decay=groups[0]["weight_decay"], betas=(0.9, 0.98))
        for g in groups[1:]:
            opt.add_param_group(g)
    else:
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    W, _, _, Y = load("eval")
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / "log.jsonl").open("a") if rank == 0 else None
    best = -1.0
    m0 = evaluate(model, W, Y, rank, world, feat=feat)
    if rank == 0:
        print(f"[train] {len(ds)} clips, {len(dl)} steps an epoch on {world} GPUs, {steps} steps, {n_ckpt} blocks "
              f"checkpointed; start mAP {m0:.4f}", flush=True)
        log.write(json.dumps({"step": 0, "map": round(m0, 4), "args": {k: str(v) for k, v in vars(a).items()}}) + "\n")
        log.flush()
    step, t0, losses = 0, time.time(), []
    rng = np.random.default_rng(a.seed + rank)
    while step < steps:
        sampler.set_epoch(step // len(dl))
        for w, y in dl:
            w, y = w.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if rng.random() < a.mixup:                       # mixed clips, their targets mixed alike
                lam = float(rng.beta(a.alpha, a.alpha))
                perm = torch.randperm(w.shape[0], device=w.device)
                w, y = lam * w + (1 - lam) * w[perm], lam * y + (1 - lam) * y[perm]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = logits_of(ddp(feat(w, train=True)))
            loss = F.binary_cross_entropy_with_logits(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            losses.append(loss.item())
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
            if step % a.eval_every == 0 or step == steps:
                m_ema = evaluate(ema, W, Y, rank, world, feat=feat)
                m_raw = evaluate(model, W, Y, rank, world, feat=feat)
                if rank == 0:
                    rec = {"step": step, "epoch": round(step / len(dl), 2), "loss": round(float(np.mean(losses)), 5),
                           "lr": opt.param_groups[0]["lr"], "map_ema": round(m_ema, 4), "map": round(m_raw, 4),
                           "min": round((time.time() - t0) / 60, 1)}
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(f"[train] {json.dumps(rec)}", flush=True)
                    if m_ema > best:
                        best = m_ema
                        torch.save({"state": ema.state_dict(), "init": base, "from": a.init, "map": m_ema,
                                    "step": step}, a.out / "best.pt")
                    losses = []
            if step >= steps:
                break
    if rank == 0:
        print(f"[train] best EMA mAP {best:.4f}", flush=True)
        print("TRAIN_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
