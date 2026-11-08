#!/usr/bin/env python3
"""Score the emotion candidates on labelled speech in many languages. RUNS ON THE BOX.

CAMEO (amu-cai/CAMEO, CC BY-NC-SA 4.0) gathers 13 emotional-speech corpora in 8 languages behind
one schema; NonverbalTTS (deepvk/NonverbalTTS, test split) adds English expressive speech from
VoxCeleb and Expresso with crowd emotion labels. Both are acted or semi-acted, so the absolute
numbers flatter every model; what this decides is which model, which combination, and which
per-label threshold Melati uses, and those are relative questions.

Reported per corpus and overall: accuracy and unweighted average recall (UAR, the SER standard).

Two things keep the numbers honest:

  * emotion2vec+ was fine-tuned on EmoBox, which contains 12 of CAMEO's 13 corpora (`SEEN`), so
    its scores on those are not evidence. HELD-OUT is the corpora no candidate is known to have
    trained on (nEMO and NonverbalTTS); the choice of model and every threshold is made there.
  * acted corpora are balanced, podcasts are mostly neutral. A threshold that gives 0.8 precision
    when a seventh of the clips are angry gives far less when one in twenty is. Each label's
    threshold is the lowest at which precision, re-weighted to the label's assumed rate in
    podcast speech (`PRIORS`, --prior-scale), reaches --target-precision on HELD-OUT. Those
    are what export_config.json's `emotion_thresholds` take.

Probabilities are saved beside the JSON (.npz), so thresholds can be re-derived without a GPU:

    python scripts/eval_emotion.py --models meralion e2v --per-corpus 300 \\
        --endpoints $MELATI_DATA/serve_eval/endpoints.json --out results/emotion/eval.json
    python scripts/eval_emotion.py --from-npz results/emotion/eval.npz --out results/emotion/eval.json

The models are asked through the tag server (serve/tag_server.py), like every other GPU call.
"""
import argparse, asyncio, io, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from melati import emotion as EMO  # noqa: E402

# corpora a candidate was trained on, from its own card (emotion2vec+: EmoBox, github.com/emo-box/EmoBox)
SEEN = {"e2v": {"cafe", "crema_d", "emns", "emozionalmente", "enterface", "jl_corpus", "mesd", "oreau",
                "pavoque", "ravdess", "resd", "subesco"}}
# assumed share of podcast segments carrying each label: an assumption, stated, not a measurement
PRIORS = {"happy": 0.15, "sad": 0.05, "angry": 0.05, "surprised": 0.05, "fearful": 0.02, "disgusted": 0.02}

MAP = {"neutral": "neutral", "calm": "neutral", "happy": "happy", "happiness": "happy", "joy": "happy",
       "sad": "sad", "sadness": "sad", "angry": "angry", "anger": "angry", "fear": "fearful", "fearful": "fearful",
       "disgust": "disgusted", "disgusted": "disgusted", "surprise": "surprised", "surprised": "surprised"}


def cameo(per_corpus, rng):
    from datasets import Audio, load_dataset
    ds = load_dataset("amu-cai/CAMEO")
    out = []
    for split in ds:
        d = ds[split].cast_column("audio", Audio(decode=False))
        rows = [(i, MAP.get(str(e).lower())) for i, e in enumerate(d["emotion"])]
        by = defaultdict(list)
        for i, lab in rows:
            if lab:
                by[lab].append(i)
        take = []
        per_lab = max(1, per_corpus // max(1, len(by)))
        for lab, idx in by.items():
            rng.shuffle(idx)
            take += [(i, lab) for i in idx[:per_lab]]
        for i, lab in take:
            r = d[i]
            out.append({"corpus": split, "language": r["language"], "label": lab, "audio": r["audio"]["bytes"]})
    return out


def nonverbaltts(rng, n=400):
    from datasets import Audio, load_dataset
    d = load_dataset("deepvk/NonverbalTTS", split="test").cast_column("audio", Audio(decode=False))
    out = []
    for r in d:
        lab = MAP.get(str(r["Emotion"]).lower())
        if lab:
            out.append({"corpus": "nonverbaltts", "language": "en", "label": lab, "audio": r["audio"]["bytes"]})
    rng.shuffle(out)
    return out[:n]


OMNI_PROMPT = ("What emotion does the speaker's voice express in this clip? Answer with exactly one word from this "
               "list: " + ", ".join(EMO.LABELS) + ".")


def omni_probs(top):
    """First-token log-probabilities -> a distribution over EMO.LABELS: each label takes the mass of the
    tokens it starts with (a label may be split, "surprised" -> "sur" + ...), renormalised."""
    p = np.zeros(len(EMO.LABELS), np.float64)
    for tok, lp in top.items():
        t = tok.strip().lower()
        if len(t) < 2:
            continue
        for k, lab in enumerate(EMO.LABELS):
            if lab.startswith(t) or t.startswith(lab):
                p[k] += np.exp(lp)
                break
    return (p / p.sum()).astype(np.float32) if p.sum() > 0 else np.full(len(EMO.LABELS), 1 / len(EMO.LABELS), np.float32)


async def score_all(args, clips, models):
    from melati.client import Client, Endpoints
    order = sorted(range(len(clips)), key=lambda j: len(clips[j]))
    chunks = [order[i:i + 32] for i in range(0, len(order), 32)]
    probs = {}
    async with Client(Endpoints(args.endpoints)) as c:
        if "omni" in models:                     # Qwen3-Omni through the vLLM `omni` service
            res = await asyncio.gather(*[c.chat_audio("omni", x, OMNI_PROMPT, 4, top_logprobs=20) for x in clips])
            probs["omni"] = np.stack([omni_probs(top) for _, top in res])
            print(f"omni: answers {Counter(t.strip().lower()[:12] for t, _ in res).most_common(9)}", flush=True)
        for name in [m for m in models if m != "omni"]:
            try:
                res = await asyncio.gather(*[c.emotion(name, [clips[j] for j in ch]) for ch in chunks])
            except ValueError as e:
                print(f"{name}: not served ({str(e)[:80]})", flush=True)
                continue
            P = np.zeros((len(clips), len(EMO.LABELS)), np.float32)
            for ch, p in zip(chunks, res):
                P[ch] = p
            probs[name] = P
    return probs


def uar(y, p):
    labs = sorted(set(y))
    return float(np.mean([np.mean([pp == l for yy, pp in zip(y, p) if yy == l]) for l in labs]))


def thresholds(P, pred, y, idx, target, prior_scale, min_sel=10):
    """Per label: the lowest threshold whose precision at the label's assumed podcast rate reaches
    `target` on clips `idx`. The false-positive rate is Laplace-smoothed, so a threshold that
    happens to select no negatives in a small set is not credited with perfect precision."""
    out = {}
    for k, lab in enumerate(EMO.LABELS):
        if lab == "neutral":
            continue
        pos = {j for j in idx if y[j] == lab}
        neg = set(idx) - pos
        best = None
        if len(pos) >= 10:
            pi = min(0.5, PRIORS[lab] * prior_scale)
            for t in np.arange(0.3, 0.96, 0.05):
                sel = {j for j in idx if pred[j] == lab and P[j, k] >= t}
                if len(sel) < min_sel:
                    break
                tpr = len(sel & pos) / len(pos)
                fpr = (len(sel & neg) + 1) / (len(neg) + 2)
                prec = tpr * pi / (tpr * pi + fpr * (1 - pi))
                if prec >= target:
                    best = {"threshold": round(float(t), 2), "precision_at_prior": round(prec, 4), "prior": pi,
                            "precision_balanced": round(len(sel & pos) / len(sel), 4), "recall": round(tpr, 4),
                            "n_pos": len(pos)}
                    break
        out[lab] = best
    return out


def curves(P, pred, y, idx, prior_scale):
    """Per label, over thresholds 0.30-0.95: [threshold, precision at the label's assumed podcast rate,
    precision on the clips as they are, recall, clips selected], on clips `idx`."""
    out = {}
    for k, lab in enumerate(EMO.LABELS):
        if lab == "neutral":
            continue
        pos = {j for j in idx if y[j] == lab}
        neg = set(idx) - pos
        if len(pos) < 10:
            continue
        pi = min(0.5, PRIORS[lab] * prior_scale)
        rows = []
        for t in np.arange(0.3, 0.96, 0.05):
            sel = {j for j in idx if pred[j] == lab and P[j, k] >= t}
            tpr = len(sel & pos) / len(pos)
            fpr = (len(sel & neg) + 1) / (len(neg) + 2)
            rows.append([round(float(t), 2), round(tpr * pi / (tpr * pi + fpr * (1 - pi)), 4) if tpr else 0.0,
                         round(len(sel & pos) / len(sel), 4) if sel else None, round(tpr, 4), len(sel)])
        out[lab] = rows
    return out


def score(probs, y, corpus, args):
    res = {"n": len(y), "corpora": dict(Counter(corpus)), "seen": {m: sorted(c) for m, c in SEEN.items()},
           "priors": PRIORS, "prior_scale": args.prior_scale, "target_precision": args.target_precision,
           "models": {}}
    seen_any = set().union(*SEEN.values())
    held = [j for j, c in enumerate(corpus) if c not in seen_any]
    res["held_out"] = sorted({corpus[j] for j in held})
    for name, P in probs.items():
        pred = [EMO.LABELS[int(k)] for k in P.argmax(1)]
        per = {}
        for c in sorted(set(corpus)):
            idx = [j for j, cc in enumerate(corpus) if cc == c]
            per[c] = {"acc": round(float(np.mean([pred[j] == y[j] for j in idx])), 4),
                      "uar": round(uar([y[j] for j in idx], [pred[j] for j in idx]), 4), "n": len(idx),
                      "seen": any(c in SEEN.get(m, ()) for m in name.split("+"))}
        unseen = [j for j, c in enumerate(corpus) if not any(c in SEEN.get(m, ()) for m in name.split("+"))]
        res["models"][name] = {
            "acc": round(float(np.mean([a == b for a, b in zip(pred, y)])), 4), "uar": round(uar(y, pred), 4),
            "held_out": {"acc": round(float(np.mean([pred[j] == y[j] for j in held])), 4),
                         "uar": round(uar([y[j] for j in held], [pred[j] for j in held]), 4), "n": len(held)},
            "unseen_by_model": {"acc": round(float(np.mean([pred[j] == y[j] for j in unseen])), 4),
                                "uar": round(uar([y[j] for j in unseen], [pred[j] for j in unseen]), 4),
                                "n": len(unseen)},
            "per_corpus": per,
            "thresholds": thresholds(P, pred, y, held, args.target_precision, args.prior_scale),
            "curves_held_out": curves(P, pred, y, held, args.prior_scale)}
        r = res["models"][name]
        print(f"{name}: all {r['acc']}/{r['uar']}  held-out {r['held_out']['acc']}/{r['held_out']['uar']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["meralion", "e2v"], help="meralion, e2v, omni")
    ap.add_argument("--combos", nargs="*", default=["meralion+e2v"], help="model averages to score too")
    ap.add_argument("--per-corpus", type=int, default=300)
    ap.add_argument("--target-precision", type=float, default=0.75,
                    help="0.75: the bar export_config.json's emotion_thresholds were chosen at")
    ap.add_argument("--prior-scale", type=float, default=1.0, help="x every PRIORS rate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--endpoints", nargs="+", type=Path, default=None, help="several files merge")
    ap.add_argument("--from-npz", type=Path, default=None, help="re-score saved probabilities, no GPU")
    args = ap.parse_args()
    probs, y, corpus = {}, None, None
    if args.from_npz:                         # saved probabilities; only models missing from it are run
        z = np.load(args.from_npz, allow_pickle=False)
        y, corpus = [str(v) for v in z["y"]], [str(v) for v in z["corpus"]]
        probs = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
    need = [m for m in args.models if m not in probs]
    if need:
        rng = random.Random(0)
        items = cameo(args.per_corpus, rng) + nonverbaltts(rng)
        clips = []
        for it in items:
            x, sr = sf.read(io.BytesIO(it["audio"]), dtype="float32", always_2d=True)
            clips.append(A.resample(x.mean(1), sr, 16000)[: 30 * 16000])
        print(f"{len(clips)} clips, {Counter(i['corpus'] for i in items)}", flush=True)
        y2, c2 = [it["label"] for it in items], [it["corpus"] for it in items]
        if y is not None and (y2 != y or c2 != corpus):
            raise SystemExit("the draw differs from the saved one: same --per-corpus and seed are required")
        y, corpus = y2, c2
        probs.update(asyncio.run(score_all(args, clips, need)))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out.with_suffix(".npz"), y=np.array(y), corpus=np.array(corpus),
                            **{f"p_{k}": v for k, v in probs.items() if "+" not in k})
    for combo in args.combos:                 # averaged probabilities of several models
        parts = combo.split("+")
        if all(p in probs for p in parts):
            probs[combo] = np.mean([probs[p] for p in parts], 0)
    res = score(probs, y, corpus, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
