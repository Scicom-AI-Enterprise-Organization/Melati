#!/usr/bin/env python3
"""A frame-level head on Melati's AudioSet tagger: where in a clip each of AudioSet's 527 sounds is, every 160 ms. One
pass of the encoder feeds the clip head (its class token, as audioset/train.py distilled it) and a frame head (each
160 ms of patch tokens, its 8 frequency bands averaged, normalised and classified, started from the clip head's
weights since the classes are the same). The loss is the clip distillation as before plus binary cross-entropy against
the strongly labelled clips' hand-timed events, frame by frame (a clip without timed events adds no frame loss).
Scored as frame mAP on the strongly labelled subset's test clips (each class's average precision over every 160 ms of
every clip, macro-averaged over the classes with a positive frame) and as clip mAP on the evaluation set. Keeps the
checkpoint with the best frame mAP (best.pt) and the last one (last.pt), which is where a run with `--frame-weight 0`
(the control on what the timed events add to the clip head) is compared. RUNS ON THE BOX, in .venv-sed, one process a
GPU:

    CUDA_VISIBLE_DEVICES=0,1,2,4,5 torchrun --nproc_per_node 5 audioset/frames_train.py --pool clean \\
        --init melati:$MELATI_DATA/models/audioset/big_from_v1/best.pt --out $MELATI_DATA/models/audioset/frames1
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, classes, load  # noqa: E402
from evaluate import mean_ap  # noqa: E402
from train import Pool, checkpointed, evaluate, features  # noqa: E402
from zoo import REPOS  # noqa: E402

STEP_S, STEPS = 0.16, 64                       # EAT's patch grid: 64 steps of 16 frames (10 ms each) over 10.24 s


def frame_targets(events, index):
    """Timed events [(onset, offset, mid)] -> (64, 527) 0/1, a step on where an event covers any of it."""
    y = np.zeros((STEPS, len(index)), np.float32)
    for a, b, mid in events:
        k = index.get(mid)
        if k is not None and b > a:
            y[int(a // STEP_S):min(STEPS, int(math.ceil(b / STEP_S))), k] = 1.0
    return y


class FramePool(Pool):
    def __init__(self, name, hard):
        super().__init__(name, hard)
        m = np.load(ROOT / "cache" / f"pool_{name}_meta.npz")
        self.strong = m["strong"]
        self.index = {mid: i for i, mid, _ in classes()}

    def __getitem__(self, i):
        w, y = super().__getitem__(i)
        ev = json.loads(self.strong[i])
        fy = frame_targets(ev, self.index) if ev is not None else np.zeros((STEPS, len(self.index)), np.float32)
        return w, y, torch.from_numpy(fy), torch.tensor(ev is not None)


class FrameEAT(nn.Module):
    def __init__(self, eat):
        super().__init__()
        self.eat = eat
        d = eat.config.embed_dim
        self.frame_norm = nn.LayerNorm(d)
        self.frame_head = nn.Linear(d, eat.config.num_classes)
        self.frame_norm.load_state_dict(eat.model.fc_norm.state_dict())     # same classes, same feature space
        self.frame_head.load_state_dict(eat.model.head.state_dict())

    def forward(self, x):
        m = self.eat.model
        z = m.encode(x)
        clip = m.head(m.fc_norm(z[:, 0]))
        fr = z[:, 1:].reshape(z.shape[0], STEPS, -1, z.shape[-1]).mean(2)    # tokens run time-major, 8 bands a step
        return clip, self.frame_head(self.frame_norm(fr))


@torch.no_grad()
def frame_map(model, W, FY, rank, world, batch=48):
    """Frame mAP on the strong test clips, each process scoring every world-th clip."""
    model.eval()
    idx = np.arange(rank, len(W), world)
    P = []
    for i in range(0, len(idx), batch):
        w = torch.from_numpy(np.asarray(W[idx[i:i + batch]], np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(torch.sigmoid(model(features(w))[1].float()).cpu().numpy().astype(np.float16))
    P = np.concatenate(P) if P else np.zeros((0, STEPS, FY.shape[-1]), np.float16)
    gathered = [None] * world
    dist.all_gather_object(gathered, (idx, P))
    model.train()
    if rank:
        return None
    full = np.zeros((len(W), STEPS, FY.shape[-1]), np.float16)
    for ix, p in gathered:
        full[ix] = p
    return mean_ap(full.reshape(-1, FY.shape[-1]).astype(np.float32), FY.reshape(-1, FY.shape[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="clean")
    ap.add_argument("--init", required=True, help="melati:<checkpoint> from audioset/train.py, or a zoo name")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5, help="the encoder's; the frame head learns at 20 times it")
    ap.add_argument("--frame-weight", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import AutoModel
    if a.init.startswith("melati:"):
        ck = torch.load(a.init.split(":", 1)[1], map_location="cpu", weights_only=False)
        eat = AutoModel.from_pretrained(REPOS[ck["init"]], trust_remote_code=True)
        eat.load_state_dict(ck["state"])
        base = ck["init"]
    else:
        eat = AutoModel.from_pretrained(REPOS[a.init], trust_remote_code=True)
        base = a.init
    model = FrameEAT(eat).cuda().train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    checkpointed(model.eat)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    ds = FramePool(a.pool, 0.0)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=a.seed, drop_last=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers, pin_memory=True,
                                     drop_last=True, persistent_workers=True)
    steps = int(a.epochs * len(dl))
    head = [p for n, p in model.named_parameters() if n.startswith("frame_")]
    enc = [p for n, p in model.named_parameters() if not n.startswith("frame_")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": 20 * a.lr}], weight_decay=a.wd,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    W, _, _, Y = load("eval")
    ST, _, _, _ = load("strong_test")
    sm = np.load(ROOT / "cache" / "strong_test_meta.npz")
    index = {mid: i for i, mid, _ in classes()}
    FY = np.stack([frame_targets(json.loads(s) or [], index) for s in sm["strong"]]).astype(np.uint8)
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / "log.jsonl").open("a") if rank == 0 else None
    best = -1.0

    def score(m, step, extra=None):
        nonlocal best
        fm, cm = frame_map(m, ST, FY, rank, world), evaluate(m, W, Y, rank, world)
        if rank == 0:
            rec = {"step": step, "frame_map": round(fm, 4), "clip_map": round(cm, 4), **(extra or {})}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[frames] {json.dumps(rec)}", flush=True)
            keep = {"state": m.state_dict(), "init": base, "from": a.init, "frame_map": fm, "clip_map": cm, "step": step}
            if step and fm > best:
                best = fm
                torch.save(keep, a.out / "best.pt")
            if step == steps:      # a run without the frame loss is compared here, not at its best frame mAP
                torch.save(keep, a.out / "last.pt")
    if rank == 0:
        print(f"[frames] {len(ds)} clips, {int(ds.strong.size - (ds.strong == 'null').sum())} with timed events, "
              f"{len(dl)} steps an epoch on {world} GPUs, {steps} steps", flush=True)
        log.write(json.dumps({"step": -1, "args": {k: str(v) for k, v in vars(a).items()}}) + "\n")
    score(model, 0)
    step, t0, losses = 0, time.time(), []
    while step < steps:
        sampler.set_epoch(step // len(dl))
        for w, y, fy, has in dl:
            w, y = w.cuda(non_blocking=True), y.cuda(non_blocking=True)
            fy, has = fy.cuda(non_blocking=True), has.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                clip, fr = ddp(features(w, train=True))
            loss = F.binary_cross_entropy_with_logits(clip.float(), y)
            if has.any():
                fl = F.binary_cross_entropy_with_logits(fr.float(), fy, reduction="none").mean((1, 2))
                loss = loss + a.frame_weight * (fl * has).sum() / has.sum()
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
                score(ema, step, {"loss": round(float(np.mean(losses)), 5), "min": round((time.time() - t0) / 60, 1)})
                losses = []
            if step >= steps:
                break
    if rank == 0:
        print(f"[frames] best frame mAP {best:.4f}", flush=True)
        print("FRAMES_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
