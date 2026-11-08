#!/usr/bin/env python3
"""Build, check and publish Melati's emotion classifier on Hugging Face. RUNS ON THE BOX.

One checkpoint (melati_hf/merge.py): the encoder of openai/whisper-large-v3 and the Whisper head trained on the
corpora whose licence explicitly allows derivatives under non-commercial research terms (deploy/box_emo_release_head.sh).
Beside it: the Whisper feature extractor's config, the helper (emotion/hf/melati_emotion.py), the licence, and a card
whose numbers and figures come from results files. Before anything is pushed, the merged model must reproduce the
pipeline's own head class (melati.ser.HeadSER) on held-out clips.

    python emotion/release.py --head $MELATI_DATA/models/ser_heads_release/whisper.pt \\
        --out $MELATI_DATA/release/Melati-Emotion-Whisper-Large-V3 \\
        --repo Scicom-intl/Melati-Emotion-Whisper-Large-V3 [--push]
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "emotion"))
from ensemble import load as load_probs, thresholds_cv  # noqa: E402
from probe import LABELS, PRIORS  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

R = Path(f"{MELATI_DATA}/results/emotion")
TRAIN = [  # corpus, what it is, licence, where
    ("emns", "EMNS, English, acted", "Apache-2.0", "https://openslr.org/136/"),
    ("emozionalmente", "Emozionalmente, Italian, crowd-acted", "CC BY 4.0", "https://zenodo.org/records/6569824"),
    ("jl_corpus", "JL-Corpus, New Zealand English, acted", "CC0", "https://www.kaggle.com/datasets/tli725/jl-corpus"),
    ("mesd", "MESD, Mexican Spanish, acted", "CC BY 4.0", "https://data.mendeley.com/datasets/cy34mh68j9/5"),
    ("oreau", "Oreau, French, acted", "CC BY 4.0", "https://zenodo.org/records/4405783"),
    ("resd", "RESD, Russian, acted dialogues", "MIT", "https://huggingface.co/datasets/Aniemore/resd_annotated"),
    ("subesco", "SUBESCO, Bangla, acted", "CC BY 4.0", "https://zenodo.org/records/4526477"),
    ("emovdb", "EmoV-DB, English, acted", "EmoV-DB non-commercial licence",
     "https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md"),
    ("expresso", "Expresso, English, read and improvised", "CC BY-NC 4.0",
     "https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset"),
]
EXCLUDED = ("CREMA-D (ODbL), RAVDESS, CaFE and PAVOQUE (CC BY-NC-SA 4.0), MELD (GPL-3.0, clipped from a TV series), "
            "NonverbalTTS's labels (CC BY-NC-SA 4.0), TurEV-DB (CC BY-NC-ND 4.0), and eNTERFACE, EmoDB, ASED, ShEMO "
            "and URDU (no licence text)")
LANG_NAMES = {"bn": "Bangla", "en": "English", "es": "Spanish", "fr": "French", "it": "Italian", "pl": "Polish",
              "ru": "Russian"}
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def pct(v, nd=0):
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def numbers():
    rel = J(R / "release/probe_whisper.json")
    full = J(R / "probe_whisper.json")
    cur = J(R / "current.json")
    name, P, Y = load_probs(str(R / "release/probs_whisper.npz"))
    ids = sorted(P)
    cv_rel = thresholds_cv(np.stack([P[i] for i in ids]), np.array([Y[i] for i in ids]), ids)
    _, Pm, Ym = load_probs(f"{R / 'current.npz'}:meralion+e2v")
    ids2 = sorted(Pm)
    cv_old = thresholds_cv(np.stack([Pm[i] for i in ids2]), np.array([Ym[i] for i in ids2]), ids2)
    uar = {"MERaLiON-SER": cur["meralion"]["held_out"]["all"]["uar"],
           "emotion2vec+ large": cur["e2v"]["held_out"]["all"]["uar"],
           "the two averaged": cur["meralion+e2v"]["held_out"]["all"]["uar"],
           "Whisper head, all 21 corpora (Melati internal)": full["held_out"]["all"]["uar"],
           "this model": rel["held_out"]["all"]["uar"]}
    return rel, full, cur, cv_rel, cv_old, uar


def figures(out: Path, rel, cv_rel, cv_old, uar):
    """The card's charts, one per file, in the preprint's style (melati/plotstyle.py)."""
    import melati.plotstyle as st
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    pct_axis = mt.PercentFormatter(1.0, decimals=0)

    def save(fig, name):
        (out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "figures" / name, dpi=st.CARD_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
    # 1. held-out UAR: the released classifiers (grey), the old tagger (amber), Melati's internal head, this model
    names = list(uar)
    cols = [st.RULE, st.RULE, st.S4, st.S5, st.S1]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 0.8))
    st.frame(ax, st.better("unweighted average recall, 7,791 held-out clips (nEMO, NonverbalTTS's VoxCeleb half)",
                           lower=False), None, axis="x")
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, [uar[n] for n in names], 0.6, color=cols, zorder=3)
    for y, n in zip(ys, names):
        ax.text(uar[n] + 0.005, y, f"{uar[n]:.3f}", va="center", fontsize=6.8, color=st.TEXT)
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=7.4)
    ax.set_xlim(0, 0.7)
    save(fig, "held_out.png")
    # 2. recall per emotion at 75% precision at podcast rates
    labs = [l for l in LABELS if l != "neutral"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    st.frame(ax, None, st.better("recall at 75% precision,\nre-weighted to podcast rates", lower=False), axis="y")
    x = np.arange(len(labs))
    series = ((cv_old, st.S4, "MERaLiON-SER + emotion2vec+, averaged"), (cv_rel, st.S1, "this model"))
    vals = [[(cv.get(l) or {}).get("recall") for l in labs] for cv, _, _ in series]
    neither = [all(vs[i] is None for vs in vals) for i in range(len(labs))]
    for off, (_, c, lab), vs in zip((-0.2, 0.2), series, vals):
        ax.bar(x + off, [v or 0 for v in vs], 0.38, color=c, zorder=3, label=lab)
        for i, v in enumerate(vs):
            if not neither[i]:      # one label for a class neither model tags, or the two collide
                ax.text(i + off, (v or 0) + 0.012, f"{100 * v:.0f}%" if v is not None else "not\ntagged",
                        ha="center", va="bottom", fontsize=6.4, color=st.TEXT)
    for i in np.flatnonzero(neither):
        ax.text(i, 0.012, "neither\ntags it", ha="center", va="bottom", fontsize=6.4, color=st.MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.yaxis.set_major_formatter(pct_axis)
    ax.set_ylim(0, 1.0)
    st.legend_below(ax, ncol=2, pad=0.14)
    save(fig, "recall.png")
    # 3. leave one corpus out
    loco = rel.get("loco") or {}
    names = [c for c, *_ in TRAIN if c in loco]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(names) + 0.8))
    st.frame(ax, st.better("UAR on each training corpus when it is left out of training", lower=False), None,
             axis="x")
    ys = np.arange(len(names))[::-1]
    ax.scatter([loco[c]["uar"] for c in names], ys, s=36, color=st.S1, zorder=4)
    for y, c in zip(ys, names):
        ax.text(loco[c]["uar"] + 0.008, y, f"{loco[c]['uar']:.2f}", va="center", fontsize=6.8, color=st.TEXT)
    ax.set_yticks(ys); ax.set_yticklabels([dict((t[0], t[1]) for t in TRAIN)[c] for c in names], fontsize=7)
    ax.set_xlim(0.3, 0.8)
    save(fig, "leave_one_out.png")


def licence_text(emovdb_licence: str):
    rows = "\n".join(f"- {n} ({l}): {u}" for _, n, l, u in TRAIN)
    return f"""Melati emotion classifier: licence

The weights in this repository may be used, reproduced, modified and redistributed for NON-COMMERCIAL RESEARCH
PURPOSES ONLY (research, teaching and publication; not commercial products, services or licensing). Any redistribution
of the weights or of works derived from them must carry this licence.

Why: the training data includes EmoV-DB, licensed for non-commercial purposes (reproduced below), and Expresso
(CC BY-NC 4.0). Every corpus used carries an explicit licence that allows derivative works under these terms:
{rows}

Excluded on licence grounds: {EXCLUDED}.

The encoder is openai/whisper-large-v3 (MIT). The software in this repository is part of Melati
(https://github.com/Scicom-AI-Enterprise-Organization/Melati).

---------------------------------------------------------------------------------------------------------------
EmoV-DB licence (https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md):

{emovdb_licence.strip()}
"""


def img(out, repo, name, alt):
    """The card's <img> for figures/<name>, at the scale every Melati card shows its figures at (melati/plotstyle.py)."""
    import melati.plotstyle as st
    return st.card_img(out / "figures" / name, alt, st.hf_url(repo, f"figures/{name}"))


def card(cfg, repo, rel, full, cur, cv_rel, cv_old, uar, out):
    ho = rel["held_out"]
    rows = "\n".join(f"| {l} | {cfg['thresholds'].get(l, 'not tagged')} | {pct(PRIORS.get(l))} | "
                     f"{pct((cv_rel.get(l) or {}).get('recall'))} | {pct((cv_rel.get(l) or {}).get('precision_at_prior'))} |"
                     for l in LABELS if l != "neutral")
    train_rows = "\n".join(f"| {n} | {rel_n(c)} | {l} |" for c, n, l, _ in TRAIN)
    return f"""---
license: other
license_name: melati-emotion-research
license_link: LICENSE
base_model: openai/whisper-large-v3
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio-classification
- speech-emotion-recognition
- emotion
- speech-data
---

# Melati emotion classifier (Whisper-large-v3)

Which of seven emotions ({", ".join(cfg['labels'])}) a stretch of speech carries. Melati's speech-data pipeline
puts emotion scopes into podcast transcripts (`<happy>...</happy>`); this is the classifier behind them that can be
released, trained only on corpora whose licence allows it.

One checkpoint: the encoder of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) (float16) and a
small head trained on its frozen features (float32): every encoder layer mean-pooled over the clip's own frames, a
learned softmax mix of the layers, LayerNorm, one 256-unit hidden layer and seven outputs. Code, training and
evaluation: [Melati on GitHub]({cfg['code']}).

## Use

```python
# pip install torch transformers huggingface_hub
import torch
from transformers import AutoFeatureExtractor, AutoModel

repo = "{repo}"
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.float16).cuda().eval()
fe = AutoFeatureExtractor.from_pretrained(repo)

f = fe([clip_16k], sampling_rate=16000, return_tensors="pt").input_features.cuda()
frames = torch.tensor([int(len(clip_16k) / 16000 * 50)])     # Whisper pads to 30 s: pool over the clip only
probs = model(f, num_frames=frames).logits.softmax(-1)      # over model.config.labels
```

`melati_emotion.py` (in this repository) applies the thresholds the way the pipeline does: the most probable emotion,
kept only if it is neutral or clears its threshold.

```python
from huggingface_hub import hf_hub_download
import importlib.util
spec = importlib.util.spec_from_file_location("melati_emotion", hf_hub_download(repo, "melati_emotion.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
em = m.Emotion.from_pretrained(repo)
em.label(em.classify([clip_16k])[0])        # "happy", ... or "neutral"
```

## Evaluation

On {ho['all']['n']:,} held-out clips that none of the models below trained on (nEMO, Polish, acted; and the VoxCeleb half of
NonverbalTTS, English interviews; emotion2vec+ was fine-tuned on EmoBox, which holds neither):

{img(out, repo, "held_out.png", "Held-out unweighted average recall")}

| held out | this model, UAR | accuracy |
|---|---|---|
| nEMO | {ho['nemo']['uar']:.3f} | {pct(ho['nemo']['acc'])} |
| NonverbalTTS, VoxCeleb half | {ho['nvtts_voxceleb']['uar']:.3f} | {pct(ho['nvtts_voxceleb']['acc'])} |
| both | {ho['all']['uar']:.3f} | {pct(ho['all']['acc'])} |

**When a label is tagged.** Acted corpora are balanced; podcasts are mostly neutral, so a threshold that is 80%
precise when a seventh of the clips are angry is far less so when one in twenty is. Each threshold is the lowest giving
75% precision once precision is re-weighted to an assumed podcast rate for that emotion; chosen on one half of the
held-out clips and measured on the other, it holds:

{img(out, repo, "recall.png", "Recall at 75% precision, re-weighted to podcast rates")}

| emotion | threshold | assumed podcast rate | recall | precision at that rate |
|---|---|---|---|---|
{rows}

Disgust has no held-out examples, so it has no threshold and is never returned.

**Across corpora.** Each training corpus scored by the head trained on the other eight (mean UAR
{rel.get('loco_mean_uar', 0):.3f}):

{img(out, repo, "leave_one_out.png", "Leave one corpus out")}

## Training data

| corpus | clips | licence |
|---|---|---|
{train_rows}

Corpora are weighted equally in the loss, and classes equally within each corpus. Melati's internal pipeline also
trains on corpora that cannot be redistributed this way ({EXCLUDED}); that head scores
{full['held_out']['all']['uar']:.3f} on the same held-out clips.

## Limitations

- The training speech is mostly acted, in {languages()}, none of them Malay. On Malaysian podcasts, where Melati runs
  it, its labels are not yet measured against human judgements, and the podcast rates behind the thresholds are
  assumptions.
- Surprise and fear are found rarely at this precision; disgust never.
- One label per clip: emotion that changes inside a clip is averaged.

## Licence

Non-commercial research only; see [LICENSE](LICENSE).

## Citation

```
@misc{{zolkepli2026melati,
  title  = {{Melati: Multilingual Expressive speech, Labelled for Affect, Tags and Interjections}},
  author = {{Husein Zolkepli}},
  year   = {{2026}},
  note   = {{Scicom (MSC) Berhad}}
}}
```
"""


COUNTS, LANGS = {}, set()


def rel_n(c):
    return f"{COUNTS.get(c, 0):,}"


def languages():
    names = sorted(LANG_NAMES.get(l, l) for l in LANGS)
    return f"{NUMBER_WORDS[len(names)]} languages ({', '.join(names[:-1])} and {names[-1]})"


def smoke(out: Path, head_dir: Path, n=64):
    """The merged model, loaded as the public loads it, against the pipeline's own head class on held-out clips."""
    sys.path.insert(0, str(out))
    from extract import Audio
    from melati.ser import HeadSER
    from melati_emotion import Emotion
    rows = [json.loads(l) for l in open(f"{MELATI_DATA}/emo/manifest_release.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "test"][:: 97][:n]
    audio = Audio()
    clips = [x[: 30 * 16000] for x in (audio(r["src"]) for r in rows) if x is not None]
    em = Emotion(out)
    a = em.classify(clips)
    b = HeadSER("whisper", heads_dir=str(head_dir))(clips)[0]
    d = float(np.abs(a - b).max())
    same = float((a.argmax(1) == b.argmax(1)).mean())
    print(f"[smoke] {len(clips)} held-out clips: max |dp| {d:.5f}, same top class {same:.3f}; labels "
          f"{[em.label(p) for p in a[:8]]}", flush=True)
    assert d < 5e-3 and same == 1.0, (d, same)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-only", action="store_true", help="upload the release already built in --out")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the card's figures into --out/figures and stop (scripts/push_card_figures.py uploads them)")
    args = ap.parse_args()
    if args.push_only:
        return push(args)
    if args.figures_only:
        rel, _, _, cv_rel, cv_old, uar = numbers()
        return figures(args.out, rel, cv_rel, cv_old, uar)
    import requests
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for line in open(f"{MELATI_DATA}/emo/manifest_release.jsonl", encoding="utf-8"):
        r = json.loads(line)
        COUNTS[r["corpus"]] = COUNTS.get(r["corpus"], 0) + 1
        if r["split"] == "train":
            LANGS.add(r["lang"])
    rel, full, cur, cv_rel, cv_old, uar = numbers()
    thresholds = {l: v["threshold"] for l, v in (rel.get("thresholds") or {}).items() if v}
    from melati_hf.merge import build_merged
    build_merged(args.head, args.out, thresholds=thresholds, task="emotion",
                 priors={l: PRIORS[l] for l in LABELS if l in PRIORS})
    cfg = json.loads((args.out / "config.json").read_text())
    cfg["code"] = "https://github.com/Scicom-AI-Enterprise-Organization/Melati"
    shutil.copy(ROOT / "emotion/hf/melati_emotion.py", args.out / "melati_emotion.py")
    emovdb = requests.get("https://raw.githubusercontent.com/numediart/EmoV-DB/master/LICENSE.md", timeout=60)
    emovdb.raise_for_status()
    (args.out / "LICENSE").write_text(licence_text(emovdb.text))
    figures(args.out, rel, cv_rel, cv_old, uar)
    (args.out / "README.md").write_text(card(cfg, args.repo, rel, full, cur, cv_rel, cv_old, uar, args.out))
    assert not re.search("—", (args.out / "README.md").read_text()), "no em dashes"
    smoke(args.out, args.head.parent)
    shutil.rmtree(args.out / "__pycache__", ignore_errors=True)
    print("files:", sorted(str(p.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file()))
    if args.push:
        push(args)
    print("RELEASE_DONE", flush=True)


def push(args):
    from huggingface_hub import HfApi
    assert (args.out / "model.safetensors").exists() and (args.out / "README.md").exists(), "build it first"
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(repo_id=args.repo, folder_path=str(args.out), repo_type="model",
                      ignore_patterns=["__pycache__/*", "*.pyc"],
                      commit_message="Melati emotion classifier: whisper-large-v3 encoder and head, one checkpoint")
    print(f"pushed https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
