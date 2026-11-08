#!/usr/bin/env python3
"""An emotion head distilled from a teacher, set against the head emotion/probe.py trains (the release's): the same
head on the same frozen features of the same labelled corpora, plus a pull towards the teacher's probabilities on
unlabelled podcast clips (emotion/pool.py, a KL term), so one model learns what several know about the speech it will
tag. Scored as probe.py scores (held-out UAR) and as the card reports (recall at 75% precision at podcast rates, each
threshold chosen on one half of the held-out clips and measured on the other: emotion/ensemble.py thresholds_cv).
RUNS ON THE BOX.

The teacher: "strong" is the seven models the pipeline averages (pool/teacher_strong.npz); "clean" is the mean of
heads trained on the licence-clean corpora alone, each on its own encoder's features of the pool (--clean-heads),
kept as pool/teacher_clean.npz; "none" (--lam 0) is the plain head, for the baseline.

    python emotion/distill.py --manifest $MELATI_DATA/emo/manifest_release.jsonl --feats $MELATI_DATA/emo/feats \\
        --pool $MELATI_DATA/emo/pool --teacher clean --clean-heads whisper=$MELATI_DATA/.../whisper.pt w2vbert=... \\
        --out $MELATI_DATA/results/emotion/distill/clean.json --save-probs ... --save-model ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensemble import thresholds_cv  # noqa: E402
from probe import LABELS, load, predict, scores, weights  # noqa: E402
from melati.ser import head_module  # noqa: E402


def pool_feats(feats, encoder, ids, corpus):
    """The pool's features for one encoder, in the order of `ids`; -> (array, the ids it holds)."""
    fdir = Path(feats) / encoder
    order = json.loads((fdir / f"{corpus}.ids.json").read_text())
    badf = fdir / f"{corpus}.bad.json"
    bad = set(json.loads(badf.read_text())) if badf.exists() else set()
    pos = {k: j for j, k in enumerate(order) if k not in bad}
    arr = np.load(fdir / f"{corpus}.npy", mmap_mode="r")
    have = [k for k in ids if k in pos]
    return np.stack([arr[pos[k]] for k in have]).astype(np.float16), have


def clean_teacher(feats, specs, ids, corpus, cache):
    """Mean of the clean heads' probabilities on the pool, each head on its own encoder's features."""
    if cache.exists():
        z = np.load(cache)
        return dict(zip(z["ids"].tolist(), z["p"]))
    per = []
    for spec in specs:
        enc, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cuda")
        head = head_module(ck["layers"], ck["dim"]).cuda()
        head.load_state_dict(ck["state"])
        head.eval()
        X, have = pool_feats(feats, enc, ids, corpus)
        per.append(dict(zip(have, predict(head, X, np.arange(len(X))))))
    common = [k for k in ids if all(k in p for p in per)]
    P = np.stack([np.mean([p[k] for p in per], 0) for k in common]).astype(np.float32)
    np.savez(cache, ids=np.array(common), p=P, members=np.array([s.split("=", 1)[0] for s in specs]))
    return dict(zip(common, P))


def train(X, y, rows, idx, Xp, Tp, lam, epochs=12, bs=512, lr=1e-3, seed=0, device="cuda"):
    """probe.train's head and schedule (balanced by corpus and class, label smoothing), plus lam times the KL
    divergence from the teacher to the head on a batch of pool clips each step."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(X[idx]).to(device)
    yt = torch.from_numpy(y[idx]).to(device)
    draw = torch.from_numpy(weights(rows, idx)).to(device)
    if lam > 0:
        Pt, Tt = torch.from_numpy(Xp).to(device), torch.from_numpy(Tp).to(device)
    head = head_module(X.shape[1], X.shape[2]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * max(1, len(idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    head.train()
    for _ in range(steps):
        b = torch.multinomial(draw, bs, replacement=True)
        loss = F.cross_entropy(head(Xt[b].float()), yt[b], label_smoothing=0.05)
        if lam > 0:
            pb = torch.randint(0, len(Pt), (bs,), device=device)
            t = Tt[pb]
            logq = F.log_softmax(head(Pt[pb].float()), -1)
            loss = loss + lam * (t * (torch.log(t + 1e-8) - logq)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    head.eval()
    return head


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--encoder", default="whisper", help="the student's encoder: the release's is whisper")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--teacher", choices=("none", "strong", "clean"), required=True)
    ap.add_argument("--clean-heads", nargs="*", default=[], metavar="ENCODER=HEAD.pt")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-probs", type=Path, default=None)
    ap.add_argument("--save-model", type=Path, default=None)
    a = ap.parse_args()
    X, rows = load(a.manifest, a.feats, a.encoder)
    y = np.array([LABELS.index(r["label"]) for r in rows])
    tr = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    test_c = sorted({r["corpus"] for r in rows if r["split"] != "train"})
    Xp = Tp = None
    n_pool = 0
    if a.teacher != "none" and a.lam > 0:
        pool_rows = [json.loads(l) for l in open(a.pool / "manifest.jsonl", encoding="utf-8")]
        ids, corpus = [r["id"] for r in pool_rows], pool_rows[0]["corpus"]
        if a.teacher == "strong":
            z = np.load(a.pool / "teacher_strong.npz")
            T = dict(zip(z["ids"].tolist(), z["p"]))
        else:
            T = clean_teacher(a.feats, a.clean_heads, ids, corpus, a.pool / "teacher_clean.npz")
        Xp, have = pool_feats(a.feats, a.encoder, [k for k in ids if k in T], corpus)
        Tp = np.stack([T[k] for k in have]).astype(np.float32)
        Tp = np.clip(Tp, 1e-6, None)
        Tp /= Tp.sum(1, keepdims=True)
        n_pool = len(have)
    head = train(X, y, rows, tr, Xp, Tp, a.lam if n_pool else 0.0, seed=a.seed)
    res = {"teacher": a.teacher, "lam": a.lam, "encoder": a.encoder, "manifest": a.manifest, "pool_clips": n_pool,
           "train_corpora": sorted({rows[i]["corpus"] for i in tr}), "held_out": {}}
    allp, ally, allid = [], [], []
    for c in test_c:
        te = np.array([i for i, r in enumerate(rows) if r["corpus"] == c])
        p = predict(head, X, te)
        res["held_out"][c] = scores(p, y[te])
        allp.append(p); ally.append(y[te]); allid += [rows[i]["id"] for i in te]
    P, Y = np.concatenate(allp), np.concatenate(ally)
    res["held_out"]["all"] = scores(P, Y)
    res["thresholds_cv"] = thresholds_cv(P, Y, allid)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    if a.save_probs:
        np.savez(a.save_probs, ids=np.array(allid), y=Y, p=P)
    if a.save_model:
        a.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": head.state_dict(), "encoder": a.encoder, "labels": LABELS, "layers": int(X.shape[1]),
                    "dim": int(X.shape[2]), "teacher": a.teacher, "lam": a.lam}, a.save_model)
    cv = res["thresholds_cv"]
    print(f"{a.out.stem}: teacher {a.teacher} lam {a.lam} pool {n_pool} | held-out UAR {res['held_out']['all']['uar']:.3f} | "
          + " ".join(f"{l} {100 * cv[l]['recall']:.1f}%" if cv.get(l) else f"{l} -" for l in LABELS if l in cv), flush=True)


if __name__ == "__main__":
    main()
