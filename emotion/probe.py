#!/usr/bin/env python3
"""Train and score an emotion head on frozen encoder features. RUNS ON THE BOX.

The head: a softmax mix of the encoder's layers, LayerNorm, a 256-unit hidden layer, seven outputs. Every corpus
counts the same in the loss (a corpus of 7,000 acted clips must not outvote one of 500), and inside a corpus every
class does.

Scores, unweighted average recall (UAR, the SER standard) and accuracy:
  loco      leave one corpus out: train on the other training corpora, test on the one left out
  held_out  train on every training corpus, test on what nothing trained on (nEMO; the VoxCeleb half of
            NonverbalTTS), with each class's threshold for 0.75 precision at podcast rates (as
            scripts/eval_emotion.py chooses the current tagger's)

    python emotion/probe.py --manifest $MELATI_DATA/emo/manifest.jsonl --feats $MELATI_DATA/emo/feats \\
        --encoder whisper --out $MELATI_DATA/results/emotion/probe_whisper.json [--save-model <path>]
"""
import argparse, json
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati.ser import LABELS, head_module  # noqa: E402   the head the tag server loads
PRIORS = {"happy": 0.15, "sad": 0.05, "angry": 0.05, "surprised": 0.05, "fearful": 0.02, "disgusted": 0.02}


def load(manifest, feats, encoder):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    fdir = Path(feats) / encoder
    X, keep = [], []
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["corpus"], []).append(i)
    for corpus, idx in by.items():
        f = fdir / f"{corpus}.npy"
        if not f.exists():
            continue
        ids = json.loads((fdir / f"{corpus}.ids.json").read_text())
        bad = set(json.loads((fdir / f"{corpus}.bad.json").read_text())) if (fdir / f"{corpus}.bad.json").exists() else set()
        pos = {k: j for j, k in enumerate(ids) if k not in bad}
        arr = np.load(f, mmap_mode="r")
        for i in idx:
            j = pos.get(rows[i]["id"])
            if j is not None:
                X.append(arr[j]); keep.append(i)
    rows = [rows[i] for i in keep]
    return np.stack(X).astype(np.float16), rows


def weights(rows, idx):
    """Each corpus the same total weight, each class the same weight inside its corpus."""
    from collections import Counter
    cc = Counter(rows[i]["corpus"] for i in idx)
    cl = Counter((rows[i]["corpus"], rows[i]["label"]) for i in idx)
    ncls = Counter(c for c, _ in cl)                                     # classes present in each corpus
    w = np.array([1.0 / (len(cc) * ncls[rows[i]["corpus"]] * cl[(rows[i]["corpus"], rows[i]["label"])]) for i in idx])
    return w / w.sum()


def train(X, y, rows, idx, epochs=12, bs=512, lr=1e-3, seed=0, device="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(X[idx]).to(device)
    yt = torch.from_numpy(y[idx]).to(device)
    w = weights(rows, idx)
    head = head_module(X.shape[1], X.shape[2]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * max(1, len(idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    draw = torch.from_numpy(w).to(device)
    head.train()
    for _ in range(steps):
        b = torch.multinomial(draw, bs, replacement=True)                 # balanced by corpus and class
        loss = nn.functional.cross_entropy(head(Xt[b].float()), yt[b], label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    head.eval()
    return head


@torch.no_grad()
def predict(head, X, idx, device="cuda", bs=2048):
    out = []
    for s in range(0, len(idx), bs):
        out.append(head(torch.from_numpy(X[idx[s:s + bs]]).to(device).float()).softmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def scores(p, y, min_n=20):
    """Accuracy, and UAR over the classes with at least `min_n` clips in this set: a class seen twice makes a
    recall of 0 or 0.5, which is noise, not evidence (NonverbalTTS's VoxCeleb half has 2 fearful clips)."""
    present = sorted(c for c in set(y.tolist()) if (y == c).sum() >= min_n)
    pred = p.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in present]
    return {"n": int(len(y)), "acc": round(float((pred == y).mean()), 4), "uar": round(float(np.mean(rec)), 4),
            "recall": {LABELS[c]: round(r, 4) for c, r in zip(present, rec)}}


def thresholds(p, y, target=0.75):
    """Per class: the lowest threshold whose precision, re-weighted to the class's podcast rate, reaches target."""
    out = {}
    for c, lab in enumerate(LABELS):
        if lab not in PRIORS or (y == c).sum() < 20:
            continue
        pos, neg = p[y == c, c], p[y != c, c]
        best = None
        for t in np.arange(0.30, 0.99, 0.01):
            tp, fp = (pos >= t).mean(), (neg >= t).mean()
            if tp == 0:
                break
            prec = PRIORS[lab] * tp / (PRIORS[lab] * tp + (1 - PRIORS[lab]) * fp + 1e-12)
            if prec >= target:
                best = {"threshold": round(float(t), 2), "recall": round(float(tp), 4),
                        "precision_at_prior": round(float(prec), 4), "n_pos": int((y == c).sum())}
                break
        out[lab] = best
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-model", type=Path, default=None)
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--save-probs", type=Path, default=None, help="held-out ids, labels and probabilities (.npz)")
    ap.add_argument("--seed", type=int, default=0, help="another seed shows how much of a difference is the shuffle")
    args = ap.parse_args()
    X, rows = load(args.manifest, args.feats, args.encoder)
    y = np.array([LABELS.index(r["label"]) for r in rows])
    corpora = sorted({r["corpus"] for r in rows})
    train_c = [c for c in corpora if all(r["split"] == "train" for r in rows if r["corpus"] == c)]
    test_c = [c for c in corpora if c not in train_c]
    print(f"{args.encoder}: {len(rows)} clips, features {X.shape[1:]}, train corpora {len(train_c)}, held out {test_c}",
          flush=True)
    res = {"encoder": args.encoder, "n": len(rows), "layers": int(X.shape[1]), "dim": int(X.shape[2]),
           "train_corpora": train_c, "held_out_corpora": test_c, "seed": args.seed, "loco": {}, "held_out": {}}
    if not args.no_loco:
        for c in train_c:
            tr = np.array([i for i, r in enumerate(rows) if r["split"] == "train" and r["corpus"] != c])
            te = np.array([i for i, r in enumerate(rows) if r["corpus"] == c])
            head = train(X, y, rows, tr, seed=args.seed)
            res["loco"][c] = scores(predict(head, X, te), y[te])
            print(f"  loco {c:16s} {res['loco'][c]['uar']:.3f} UAR  {res['loco'][c]['acc']:.3f} acc", flush=True)
        res["loco_mean_uar"] = round(float(np.mean([v["uar"] for v in res["loco"].values()])), 4)
    tr = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    head = train(X, y, rows, tr, seed=args.seed)
    allp, ally, allid = [], [], []
    for c in test_c:
        te = np.array([i for i, r in enumerate(rows) if r["corpus"] == c])
        p = predict(head, X, te)
        res["held_out"][c] = scores(p, y[te])
        allp.append(p); ally.append(y[te]); allid += [rows[i]["id"] for i in te]
        print(f"  held out {c:16s} {res['held_out'][c]['uar']:.3f} UAR  {res['held_out'][c]['acc']:.3f} acc", flush=True)
    if allp:
        P, Y = np.concatenate(allp), np.concatenate(ally)
        res["held_out"]["all"] = scores(P, Y)
        res["thresholds"] = thresholds(P, Y)
        res["layer_mix"] = [round(float(v), 4) for v in head.mix.softmax(0).detach().cpu()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    if args.save_probs and allp:
        np.savez(args.save_probs, ids=np.array(allid), y=np.concatenate(ally), p=np.concatenate(allp))
    if args.save_model:
        torch.save({"state": head.state_dict(), "encoder": args.encoder, "labels": LABELS,
                    "layers": int(X.shape[1]), "dim": int(X.shape[2])}, args.save_model)
    print(json.dumps({k: res[k] for k in ("loco_mean_uar",) if k in res} | {"held_out": res["held_out"].get("all"),
                     "thresholds": res.get("thresholds")}), flush=True)


if __name__ == "__main__":
    main()
