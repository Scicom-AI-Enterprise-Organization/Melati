#!/usr/bin/env python3
"""Train one non-verbal head on frozen encoder features, and score it on clips nothing trained on. RUNS ON THE BOX.

The head is the emotion heads' (melati.ser.head_module: a softmax mix of the encoder's layers, LayerNorm, one
256-unit hidden layer) with an output per melati.nv_gaps.HEAD_LABELS. Features come from emotion/extract.py over
nonverbal/build_manifest.py's clips. Every corpus counts the same in the loss and, inside a corpus, every class,
so 29k fillers do not drown 5.8k laughs and VocalSound's sneezes are not outvoted by PodcastFillers' words.

Trained on the `train` splits; `validation` picks the number of epochs; `test` (PodcastFillers' test episodes, by
show; VocalSound's test speakers) is reported per class: precision, recall and F1 at the argmax, and average
precision. Saves the head the tag server loads (melati.ser.HeadNV) and the test probabilities.

    python nonverbal/train.py --manifest $MELATI_DATA/nv/manifest.jsonl --feats $MELATI_DATA/nv/feats \\
        --encoder whisper --out $MELATI_DATA/results/nonverbal/head_whisper.json \\
        --save-model $MELATI_DATA/models/nv_heads/whisper.pt
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "emotion"))
from melati.nv_gaps import HEAD_LABELS  # noqa: E402
from melati.ser import head_module  # noqa: E402
from probe import load, predict  # noqa: E402   the emotion probe's feature loader


def weights(rows, idx, min_n=30, pool_weights=None):
    """Each corpus the same total weight, each class the same weight inside its corpus, as the emotion probe
    weighs them; but a class with fewer than min_n clips in a corpus is left out of that corpus, or a handful of
    NonverbalTTS sneezes would be drawn as often as thousands of breaths. Corpora that share a `pool` count as one
    (COUGHVID and Coswara: two sets of phone-recorded coughs), and pool_weights scales a pool against the rest (a
    pool of one class otherwise gets a whole corpus's share of the loss for that class)."""
    pool_weights = pool_weights or {}
    key = [rows[i].get("pool") or rows[i]["corpus"] for i in idx]
    cl = Counter((k, rows[i]["label"]) for k, i in zip(key, idx))
    ok = {k for k, n in cl.items() if n >= min_n}
    cc = Counter(c for c, _ in ok)
    w = np.array([pool_weights.get(k, 1.0) / (cc[k] * cl[(k, rows[i]["label"])]) if (k, rows[i]["label"]) in ok
                  else 0.0 for k, i in zip(key, idx)])
    return w / w.sum()


def fit(X, y, rows, idx, epochs, bs=512, lr=1e-3, seed=0, device="cuda", val=None, pool_weights=None):
    """-> (head, [(epoch, validation macro F1)]); the head returned is the best epoch's on `val`."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xt, yt = torch.from_numpy(X[idx]).to(device), torch.from_numpy(y[idx]).to(device)
    draw = torch.from_numpy(weights(rows, idx, pool_weights=pool_weights)).to(device)
    head = head_module(X.shape[1], X.shape[2], n=len(HEAD_LABELS)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    per = max(1, len(idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per, pct_start=0.1)
    best, curve = None, []
    for ep in range(epochs):
        head.train()
        for _ in range(per):
            b = torch.multinomial(draw, bs, replacement=True)
            loss = nn.functional.cross_entropy(head(Xt[b].float()), yt[b], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        head.eval()
        if val is not None and len(val):
            with torch.no_grad():
                f1 = report(predict(head, X, val), y[val])["macro_f1"]
            curve.append((ep + 1, round(f1, 4)))
            if best is None or f1 > best[0]:
                best = (f1, {k: v.detach().clone() for k, v in head.state_dict().items()})
    if best is not None:
        head.load_state_dict(best[1])
    return head, curve


def report(p, y):
    """Per class at the argmax: precision, recall, F1, and average precision of the class's own probability."""
    from sklearn.metrics import average_precision_score
    pred = p.argmax(1)
    out, f1s = {}, []
    for k, lab in enumerate(HEAD_LABELS):
        n = int((y == k).sum())
        if not n:
            continue
        tp, hyp = int(((pred == k) & (y == k)).sum()), int((pred == k).sum())
        pr, rc = tp / hyp if hyp else 0.0, tp / n
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        f1s.append(f1)
        out[lab] = {"n": n, "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4),
                    "ap": round(float(average_precision_score(y == k, p[:, k])), 4)}
    return {"per_class": out, "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "accuracy": round(float((pred == y).mean()), 4), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-model", type=Path, default=None)
    ap.add_argument("--save-probs", type=Path, default=None, help="test ids, labels and probabilities (.npz)")
    ap.add_argument("--pool-weights", default="{}", help='JSON, a pool or corpus -> its share against the rest\'s '
                                                        '1.0 (e.g. \'{"coughs": 0.33}\')')
    args = ap.parse_args()
    args.pool_weights = json.loads(args.pool_weights)
    X, rows = load(args.manifest, args.feats, args.encoder)
    y = np.array([HEAD_LABELS.index(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "validation", "test"))
    print(f"[{args.encoder}] {X.shape} train {len(tr)} validation {len(va)} test {len(te)}", flush=True)
    head, curve = fit(X, y, rows, tr, args.epochs, val=va, pool_weights=args.pool_weights)
    with torch.no_grad():
        p = predict(head, X, te)
    res = {"encoder": args.encoder, "labels": list(HEAD_LABELS), "features": list(X.shape[1:]),
           "train": dict(Counter(f"{rows[i]['corpus']}/{rows[i]['label']}" for i in tr)),
           "pool_weights": args.pool_weights,
           "validation_curve": curve, "test": {"all": report(p, y[te])}}
    for c in sorted({rows[i]["corpus"] for i in te}):
        m = np.array([rows[i]["corpus"] == c for i in te])
        res["test"][c] = report(p[m], y[te][m])
    mix = head.mix.softmax(0).detach().cpu().numpy()
    res["layer_mix"] = [round(float(v), 4) for v in mix]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"labels": list(HEAD_LABELS), "layers": X.shape[1], "dim": X.shape[2],
                    "state": {k: v.cpu() for k, v in head.state_dict().items()}, "encoder": args.encoder,
                    "trained_on": res["train"]}, args.save_model)
    if args.save_probs:
        np.savez(args.save_probs, ids=np.array([rows[i]["id"] for i in te]), y=y[te], p=p)
    for scope, r in res["test"].items():
        print(f"[{args.encoder}] test {scope}: macro F1 {r['macro_f1']:.3f}  acc {r['accuracy']:.3f}  " +
              "  ".join(f"{k} P{v['precision']:.2f}/R{v['recall']:.2f}" for k, v in r["per_class"].items()), flush=True)
    print(f"TRAIN_DONE {args.encoder}", flush=True)


if __name__ == "__main__":
    main()
