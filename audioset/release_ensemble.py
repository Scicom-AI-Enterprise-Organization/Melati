#!/usr/bin/env python3
"""Build, check and publish Melati-AudioSet-Ensemble: six AudioSet taggers run on the same clip, their probabilities
averaged (CED-base, EAT-large, SSLAM and BEATs, released by others, and Melati-AudioSet-EAT-Large's v3 and v4), the
best average audioset/evaluate.py scored (results/audioset/teachers_beats_fp32.json). RUNS ON THE BOX, in .venv-sed.

The repo holds the code that runs them (audioset/hub/melati_ensemble.py) and the revision of each member's own repo,
plus BEATs, which has no Hugging Face home of its own: its weights (the checkpoint deploy/box_beats_fetch.sh fetched,
as safetensors) and microsoft/unilm's code for it (MIT), its imports of itself made relative. It is uploaded to a
private repo and loaded back from there as its card says, then scored on the evaluation set in fp32 and in bf16: each
member must give the probabilities audioset/evaluate.py cached for it, and the average the mAP the results file says.
Every number on the card comes from that check (results/audioset/ensemble_fp32.json). The card's Use snippet is then
run as a user would (audioset/check_card_use.py), and only then is the repo added to the Melati collection and made
public. --card-only writes the card and figures again from the last check's results, once the files that check
scored are confirmed unchanged on the Hub, and goes on from there.

    python audioset/release_ensemble.py --out $MELATI_DATA/release/Melati-AudioSet-Ensemble \\
        --repo Scicom-intl/Melati-AudioSet-Ensemble [--push | --card-only]
"""
import os
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from data import classes, load  # noqa: E402
from evaluate import PROBS, mean_ap  # noqa: E402
from release import CODE, RESULTS, labels, snippet  # noqa: E402
from zoo import BEATS  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh
MELATI_ROOT = os.environ.get("MELATI_ROOT", "/root/melati")        # environments, checkouts: deploy/env.sh

M = f"{MELATI_DATA}/models/audioset"
MELATI = "Scicom-intl/Melati-AudioSet-EAT-Large"
# each member: how ensemble.json loads it, the audioset/zoo.py name its scores are filed under, and how the card
# names it. v3 is the last commit on Melati-AudioSet-EAT-Large with v3's weights (its per-label card), v4 its release.
MEMBERS = [
    {"name": "ced-base", "kind": "ced", "repo": "mispeech/ced-base",
     "revision": "db3e14a8db4c21b56b165261c39649741a900e7f", "zoo": "ced-base", "title": "CED-base",
     "cite": "Dinkel et al., ICASSP 2024", "licence": "Apache-2.0"},
    {"name": "eat-large", "kind": "eat", "repo": "worstchan/EAT-large_epoch20_finetune_AS2M",
     "revision": "53b1251196e3dabb991dfa7629fa89394c7f8335", "zoo": "eat-large", "title": "EAT-large",
     "cite": "Chen et al., IJCAI 2024", "licence": "MIT"},
    {"name": "sslam", "kind": "eat", "repo": "ta012/SSLAM_AS2M_Finetuned",
     "revision": "4500c5d989436b57b0621418fa49a54ca2ed8dff", "zoo": "sslam", "title": "SSLAM",
     "cite": "Alex et al., ICLR 2025", "licence": "MIT"},
    {"name": "beats", "kind": "beats", "zoo": "beats", "title": "BEATs", "cite": "Chen et al., ICML 2023",
     "licence": "MIT"},
    {"name": "melati-v3", "kind": "eat", "repo": MELATI, "revision": "06a9aef399e54c99e19094f8760cdb673a6cbbd4",
     "zoo": f"melati:{M}/frames1/clip.pt", "title": "Melati-AudioSet-EAT-Large v3", "cite": "ours", "licence": "MIT"},
    {"name": "melati-v4", "kind": "eat", "repo": MELATI, "revision": "78248ff107120278b14465c301a20b57223d6b28",
     "zoo": f"melati:{M}/frames1_fw0/clip.pt", "title": "Melati-AudioSet-EAT-Large v4", "cite": "ours",
     "licence": "MIT"},
]
BASE3 = ("ced-base", "eat-large", "sslam")          # the teachers Melati-AudioSet-EAT-Large v4 was distilled from
TEACHERS = RESULTS / "teachers_beats_fp32.json"     # evaluate.py --fp32 --add-to: every member alone and averaged
OUT = RESULTS / "ensemble_fp32.json"
UNILM = Path(f"{MELATI_ROOT}/ext/beats")              # deploy/box_beats_fetch.sh: microsoft/unilm's beats/ folder
COLLECTION = "Scicom-intl/melati-6adb680852d20fe5ea6a77ae"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build(out):
    from safetensors.torch import save_file
    if out.exists():
        shutil.rmtree(out)
    (out / "melati_beats").mkdir(parents=True)
    shutil.copy(HERE / "hub" / "melati_ensemble.py", out / "melati_ensemble.py")
    shutil.copy(HERE / "hub" / "melati_engine.py", out / "melati_engine.py")
    (out / "labels.json").write_text(json.dumps([{"index": i, "mid": mid, "name": n} for i, mid, n in classes()],
                                                indent=1))
    keep = ("name", "kind", "repo", "revision")
    (out / "ensemble.json").write_text(json.dumps(
        {"sample_rate": 16000, "clip_s": 10.0, "members": [{k: m[k] for k in keep if k in m} for m in MEMBERS]},
        indent=1))
    lazy = "        import torchaudio.compliance.kaldi as ta_kaldi    # melati: here, so the package needs no torchaudio\n"
    for f, edits in (("BEATs.py", [("from backbone import", "from .backbone import"),
                                   ("import torchaudio.compliance.kaldi as ta_kaldi\n", ""),
                                   ("        fbanks = []\n", lazy + "        fbanks = []\n")]),
                     ("backbone.py", [("from modules import", "from .modules import")]),
                     ("modules.py", [])):
        src = (UNILM / f).read_text()
        for old, new in edits:
            assert src.count(old) == 1, (f, old)
            src = src.replace(old, new)
        (out / "melati_beats" / f).write_text(src)
    shutil.copy(UNILM / "LICENSE", out / "melati_beats" / "LICENSE")
    (out / "melati_beats" / "__init__.py").write_text(
        '"""BEATs (Chen et al., ICML 2023): microsoft/unilm\'s beats/BEATs.py, backbone.py and modules.py, MIT '
        '(LICENSE),\ntheir imports of each other made relative so the folder is a package, and BEATs.py\'s torchaudio '
        'import moved into\npreprocess(), the one method that uses it (melati_ensemble.py computes those filterbanks '
        'itself). Nothing else\nis changed."""\n')
    ck = torch.load(BEATS, map_location="cpu", weights_only=True)
    by = {}
    for i, mid, n in classes():
        by[mid], by[n] = i, i
    order = [by[ck["label_dict"][k]] for k in range(len(ck["label_dict"]))]
    assert sorted(order) == list(range(527)), "BEATs's labels are not AudioSet's 527"
    save_file({k: v.detach().clone().contiguous() for k, v in ck["model"].items()}, str(out / "beats.safetensors"))
    (out / "beats.json").write_text(json.dumps({"checkpoint": BEATS.name, "sha256": sha256(BEATS), "cfg": ck["cfg"],
                                                "order": order}, indent=1))


def check(repo, revision, batch):
    """The uploaded repo loaded from the Hub as its card says and run on the evaluation set: each member's
    probabilities against the ones audioset/evaluate.py cached for it, the average's mAP against the results file's,
    in fp32, then the average in bf16, and a long recording through windows(). -> the card's numbers."""
    from importlib.metadata import version
    from huggingface_hub import snapshot_download
    from sklearn.metrics import average_precision_score
    path = snapshot_download(repo, revision=revision)
    sys.path.insert(0, path)
    from melati_ensemble import Ensemble
    teachers = json.loads(TEACHERS.read_text())
    expect = teachers["averages"][" + ".join(sorted(m["zoo"] for m in MEMBERS))]
    W, _, _, Y = load("eval")
    model = Ensemble(path, device="cuda")
    assert list(model.members) == [m["name"] for m in MEMBERS]
    cached = {m["name"]: np.load(PROBS / f"eval_{m['zoo'].replace('melati:', 'melati_').replace('/', '_')}.fp32.npy")
              .astype(np.float32) for m in MEMBERS}
    per, t0 = {k: [] for k in model.members}, time.time()
    for i in range(0, len(W), batch):
        _, each = model([w.astype(np.float32) for w in W[i:i + batch]], batch=batch, each=True)
        for k, p in each.items():
            per[k].append(p)
        if i == 0:                  # fail fast: the first batch already agrees with the cache, CED-base aside
            for k, p in each.items():
                d = float(np.abs(p - cached[k][:len(p)]).max())
                print(f"[check] first batch, {k}: max |p - cached| {d:.5f}", flush=True)
                assert k == "ced-base" or d < 0.01, (k, d)
        if (i // batch) % 200 == 0:
            print(f"[check] fp32 {i + batch:,}/{len(W):,} clips, {(i + batch) / (time.time() - t0):.1f}/s", flush=True)
    fp32_s = time.time() - t0
    P = {k: np.concatenate(v) for k, v in per.items()}
    avg = np.mean([P[m["name"]] for m in MEMBERS], 0)
    got = mean_ap(avg, Y)
    print(f"[check] the ensemble from the Hub: mAP {got:.4f} (results file {expect:.4f})", flush=True)
    members = {}
    for m in MEMBERS:
        k, d = m["name"], np.abs(P[m["name"]] - cached[m["name"]])
        members[k] = {"map": round(mean_ap(P[k], Y), 4), "results_map": teachers["models"][m["zoo"]],
                      "cache_max_abs_diff": round(float(d.max()), 5), "cache_mean_abs_diff": float(d.mean()),
                      "params": int(sum(p.numel() for p in model.members[k].net.parameters()))}
        print(f"[check] {k}: {members[k]}", flush=True)
        assert abs(members[k]["map"] - members[k]["results_map"]) < 1e-3, (k, members[k])
        assert m["kind"] == "ced" or d.max() < 0.01, (k, float(d.max()))
    assert abs(got - expect) < 1e-3, (got, expect)
    names = [m["name"] for m in MEMBERS]

    def avg_map(ks):
        return round(mean_ap(np.mean([P[k] for k in ks], 0), Y), 4)
    without = {k: avg_map([n for n in names if n != k]) for k in names}
    steps = {"base3": avg_map(BASE3), "base3_v3_v4": avg_map(BASE3 + ("melati-v3", "melati-v4")),
             "base3_beats": avg_map(BASE3 + ("beats",)), "all": round(got, 4)}
    pos = Y.sum(0).astype(int)
    aps = [float(average_precision_score(Y[:, c], avg[:, c])) if pos[c] else None for c in range(Y.shape[1])]
    assert abs(np.mean([x for x in aps if x is not None]) - got) < 1e-9
    model.bf16, t0 = True, time.time()
    Pb = np.concatenate([model([w.astype(np.float32) for w in W[i:i + batch]], batch=batch)
                         for i in range(0, len(W), batch)])
    bf16_s = time.time() - t0
    got_bf16 = mean_ap(Pb, Y)
    print(f"[check] bf16: mAP {got_bf16:.4f}, {len(W) / bf16_s:.1f} clips/s (fp32 {len(W) / fp32_s:.1f})", flush=True)
    model.bf16 = False
    long = np.concatenate([W[0], W[1], W[2][:80000]]).astype(np.float32)        # 25 s: windows at 0, 5, 10, 15 s
    starts, pw = model.windows(long, hop_s=5.0)
    assert starts.tolist() == [0.0, 5.0, 10.0, 15.0] and pw.shape == (4, 527), (starts, pw.shape)
    assert np.abs(pw[0] - avg[0]).max() < 1e-3, "the first window is clip 0"
    return {"repo": repo, "revision": revision, "split": "eval", "clips": int(len(Y)),
            "classes_scored": int((pos > 0).sum()), "batch": batch, "map": round(got, 4), "expected": expect,
            "map_bf16": round(got_bf16, 4), "clips_per_s": round(len(W) / fp32_s, 1),
            "clips_per_s_bf16": round(len(W) / bf16_s, 1),
            "params": int(sum(v["params"] for v in members.values())), "members": members, "without": without,
            "steps": steps, "versions": {p: version(p) for p in ("torch", "numpy", "huggingface_hub", "safetensors")},
            "classes": [{"index": i, "mid": mid, "name": n, "ap": None if aps[i] is None else round(aps[i], 4),
                         "positives": int(pos[i])} for i, mid, n in classes()]}


def figures(out, repo, res):
    import matplotlib.pyplot as plt
    import melati.plotstyle as st
    (out / "figures").mkdir(exist_ok=True)
    title = {m["name"]: m["title"] for m in MEMBERS}

    def save(fig, name):
        fig.savefig(out / "figures" / name, dpi=st.CARD_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
    # 1. each member alone, and the ensemble, on the same clips: a dot each
    rows = sorted([(title[k], v["map"], False) for k, v in res["members"].items()]
                  + [("this ensemble (all six)", res["map"], True)], key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    st.frame(ax, st.better("mAP on AudioSet's evaluation set", lower=False), None, axis="x")
    lo = 0.47
    for i, (name, v, ours) in enumerate(rows):
        ax.plot([lo, v], [i, i], color=st.RULE, lw=0.8, zorder=1)
        ax.scatter(v, i, s=46 if ours else 32, color=st.S1 if ours else st.MUTED, zorder=3)
        ax.text(v + 0.0012, i, f"{v:.4f}", va="center", fontsize=7, color=st.TEXT, fontweight="bold" if ours else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    ax.set_xlim(lo, max(r[1] for r in rows) + 0.008)
    save(fig, "map.png")
    # 2. what each member adds: the ensemble's mAP less the other five's average
    adds = sorted(((title[k], res["map"] - v) for k, v in res["without"].items()), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    st.frame(ax, st.better("mAP the ensemble loses without it", lower=False, word="adds more"), None, axis="x")
    ax.barh(range(len(adds)), [v for _, v in adds], height=0.55, color=st.S2, zorder=3)
    top = max(v for _, v in adds)
    for i, (_, v) in enumerate(adds):
        ax.text(max(v, 0) + top * 0.02, i, f"{v:+.4f}", va="center", fontsize=7, color=st.TEXT)
    ax.axvline(0, color=st.RULE, lw=0.8, zorder=2)
    ax.set_yticks(range(len(adds)))
    ax.set_yticklabels([n for n, _ in adds], fontsize=7.6)
    ax.set_xlim(min(0, min(v for _, v in adds)) - top * 0.05, top * 1.25)
    save(fig, "members.png")
    return {n: st.card_img(out / "figures" / n, alt, st.hf_url(repo, f"figures/{n}"))
            for n, alt in (("map.png", "mAP on AudioSet's evaluation set"),
                           ("members.png", "What each member adds"))}


PIP = "pip install torch numpy huggingface_hub safetensors soundfile"
USE = """```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("REPO_ID")
sys.path.insert(0, path)
from melati_ensemble import Ensemble, read

model = Ensemble(path)             # the six members, only their weights fetched, each at a pinned revision; the GPU if any

probs = model([read("clip.wav")])[0]    # any sample rate or channels, read as 16 kHz mono; AudioSet's 527 classes
for i in probs.argsort()[::-1][:5]:
    print(model.labels[i], round(float(probs[i]), 3))
```"""
NATIVE = RESULTS / "ensemble_native.json"           # audioset/check_native.py: the members against their own code
FRESH = RESULTS / "ensemble_fresh.json"             # audioset/check_ensemble_fresh.py: a new venv, an empty cache
ENGINE = RESULTS / "engine.json"                    # audioset/check_engine.py: melati_engine.py from the Hub, measured
PIP_ENGINE = PIP + " fastapi uvicorn"


def card(repo, res, img):
    t = {m["name"]: m for m in MEMBERS}
    mem, v = res["members"], res["versions"]
    nat, fresh = json.loads(NATIVE.read_text()), json.loads(FRESH.read_text())
    assert fresh["revision"] == res["revision"] and not fresh.get("error") and all(
        m["ok"] for m in fresh["members"].values()), "the new-environment check is not of this revision, or failed"
    assert not fresh["versions"].get("transformers") and not fresh["versions"].get("torchaudio"), fresh["versions"]
    fv = fresh["versions"]
    parity = max(m["max_abs_diff"] for m in nat["compare"]["fp32"]["members"].values())
    eng = json.loads(ENGINE.read_text())
    assert eng["revision"] == res["revision"], "the engine's check is not of this revision"
    assert fresh["engine"]["ok"], "the engine did not pass in the new environment"
    assert abs(eng["eval"]["map"] - res["map_bf16"]) <= 1e-4, (eng["eval"]["map"], res["map_bf16"])
    one, eager, ev = eng["one_gpu_graphs"], eng["one_gpu_eager"], eng["eval"]
    lv = {x["concurrency"]: x for x in eng["http"]["levels"]}
    alone, some, many = lv[1], lv[16], lv[max(lv)]

    def link(k):
        m = t[k]
        return f"[{m['title']}](https://huggingface.co/{MELATI}/tree/{m['revision']})" if m.get("repo") == MELATI \
            else m["title"]
    order = sorted(mem, key=lambda k: -mem[k]["map"])
    rows = "\n".join(f"| {link(k)} | {mem[k]['map']:.4f} | {t[k]['cite']} |" for k in order)
    where = {m["name"]: (f"[{m['repo']}](https://huggingface.co/{m['repo']}/tree/{m['revision']}) at `{m['revision'][:7]}`"
                         if "repo" in m else "this repo: `beats.safetensors` (microsoft/unilm's iter3+ checkpoint "
                         "fine-tuned on AudioSet-2M, cpt2) and `melati_beats/`") for m in MEMBERS}
    members = "\n".join(f"| {t[k]['title']} ({t[k]['cite']}) | {where[k]} | {t[k]['licence']} | "
                        f"{mem[k]['params'] / 1e6:.0f}M | {mem[k]['map']:.4f} |" for k in order)
    adds = {k: res["map"] - x for k, x in res["without"].items()}
    most = max(adds, key=adds.get)
    assert max(adds["melati-v3"], adds["melati-v4"]) < 1e-3, "v3 and v4 each add little once the other is in"
    assert min(mem, key=lambda k: mem[k]["map"]) == "beats", "BEATs is the weakest member alone"
    if most == "beats":
        misses = (f"It misses BEATs most ({adds['beats']:+.4f}), the weakest member alone ({mem['beats']['map']:.4f}): "
                  "it was pre-trained differently from the rest.")
    else:
        misses = (f"It misses {t[most]['title']} most ({adds[most]:+.4f}); BEATs, the weakest member alone "
                  f"({mem['beats']['map']:.4f}), still adds {adds['beats']:+.4f}, as it was pre-trained differently "
                  "from the rest.")
    idle = sorted((k for k in adds if adds[k] <= 0), key=adds.get)       # members the rest already cover
    idle_txt = (" " + " and ".join(t[k]["title"] for k in idle) + (" adds" if len(idle) == 1 else " add")
                + " nothing on top of the rest (" + ", ".join(f"{adds[k]:+.4f}" for k in idle) + ")"
                + ("; v3 and v4 were trained from it." if idle == ["eat-large"] else ".")) if idle else ""
    s = res["steps"]
    drift = max(abs(mem[k]["map"] - mem[k]["results_map"]) for k in mem if k != "ced-base")
    assert drift <= 1e-4 + 1e-9, drift
    other_card = (f" CED-base reads {mem['ced-base']['map']:.4f} here and {mem['ced-base']['results_map']:.4f} on "
                  f"Melati-AudioSet-EAT-Large's card, where it was given 64 clips at a time (see Use)"
                  + ("; the other numbers there differ from these by 0.0001 at most (that card scores probabilities "
                     "stored in 16 bits)." if drift else "."))
    return f"""---
license: mit
pipeline_tag: audio-classification
tags:
- audio
- audio-classification
- audioset
- ensemble
base_model:
- mispeech/ced-base
- worstchan/EAT-large_epoch20_finetune_AS2M
- ta012/SSLAM_AS2M_Finetuned
- {MELATI}
---

# Melati-AudioSet-Ensemble

Six AudioSet taggers run on the same clip, their probabilities averaged: **{res['map']:.4f} mAP** on AudioSet's
evaluation set, ahead of every single model we scored on the same clips, the best of them our own
[Melati-AudioSet-EAT-Large](https://huggingface.co/{MELATI}) ({mem['melati-v4']['map']:.4f}). Four members are released by
others (CED-base, EAT-large, SSLAM and BEATs) and two are Melati-AudioSet-EAT-Large's v3 and v4. It names which of
AudioSet's [527 sounds](#labels) are in a clip, for six forward passes ({res['params'] / 1e9:.2f}B parameters in all);
Melati-AudioSet-EAT-Large is one model of {mem['melati-v4']['params'] / 1e6:.0f}M, distilled from three of these members'
average. Part of [Melati]({CODE}), Scicom's speech-data pipeline.

{img['map.png']}

| model | mAP | |
|---|---|---|
| **this ensemble** | **{res['map']:.4f}** | the six below, averaged |
{rows}

All on the same {res['clips']:,} clips: AudioSet's evaluation set as
[agkphysics/AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) holds it today (YouTube removes clips over
time, so every paper's copy differs and published numbers are not comparable across papers), each model in fp32 with
its own preprocessing, mAP the mean over the 527 classes of their average precision. Every number here is this repo's,
loaded from the Hub as the Use section loads it.{other_card}

## Use

```bash
{PIP}
```

{USE.replace("REPO_ID", repo)}

A clip is 16 kHz mono, cut or zero-padded to 10 s, the length every member was scored on. A longer recording goes
through `model.windows(wav, hop_s=5.0)`, which gives each 10 s window's start in seconds and its probabilities; the
highest over the windows tags the whole recording. `model(clips, each=True)` also returns every member's own
probabilities, and `Ensemble(path, members=[...])` runs only some of them (their names are in `ensemble.json`).

Every member is built in `melati_ensemble.py` in plain PyTorch, and only its `config.json` and `model.safetensors` are
fetched from its repo at the pinned revision: no repo's own code is run, so it needs neither transformers nor timm nor
torchaudio (soundfile only for `read`). Each member is fed its own preprocessing, as it was scored, computed for the
whole batch on the GPU. On {nat['clips_compared']} evaluation clips, each member's probabilities match its own repo's
code to within {parity:.0e} in fp32. CED-base's decibel floor is set under each clip's own loudest value, as its feature
extractor sets it for a clip given alone, so a clip's tags never depend on the clips batched with it. Revisions before
this one ran each member's own code through transformers' `trust_remote_code`, which needed timm and did not load
without it.

On one GPU, in batches of {res['batch']}, it tags {res['clips_per_s']:.0f} ten-second clips a second in fp32;
`Ensemble(path, bf16=True)` tags {res['clips_per_s_bf16']:.0f} a second at {res['map_bf16']:.4f} mAP. Tested with torch
{v['torch'].split('+')[0]}, numpy {v['numpy']}, huggingface_hub {v['huggingface_hub']} and safetensors {v['safetensors']},
and loaded into an empty cache in a new environment holding only the five packages above, at torch
{fv['torch'].split('+')[0]}, numpy {fv['numpy']}, huggingface_hub {fv['huggingface_hub']}, safetensors {fv['safetensors']}
and soundfile {fv['soundfile']}.

## Inference engine

`melati_engine.py` serves the ensemble on CUDA GPUs: every GPU holds all six members, their weights in bf16, and runs a
CUDA graph captured for each batch size, and the 10 s windows of every request share one queue, so concurrent
requests fill the same batches and every GPU takes its share. A recording longer than 10 s is cut into 10 s windows
every `hop_s` seconds (5 by default), and a sound's probability in it is its highest over the windows.

```bash
{PIP_ENGINE}
hf download {repo} --local-dir melati-ensemble
python melati-ensemble/melati_engine.py serve --gpus 0,1 --port 8000
curl -s --data-binary @clip.wav "localhost:8000/tag?top_k=5"
python melati-ensemble/melati_engine.py tag clips/ --gpus 0,1 --out tags.jsonl    # every audio file under clips/
```

`POST /tag` takes an audio file as its body (anything soundfile reads), or JSON `{{"pcm16": base64, "sample_rate":
16000}}`, and answers the `top_k` most probable sounds (`threshold` sets a floor); `windows=true` adds each window's,
`members=true` each member's and `probs=true` all 527. In Python, `Engine(gpus=[0]).tag(wave)` gives each window's
start and probabilities. `GET /stats` says how full its batches are and how busy its GPUs.

Through the engine on {len(ev['gpus'])} GPUs the evaluation set scores {ev['map']:.4f} mAP (`Ensemble(path, bf16=True)`:
{res['map_bf16']:.4f}), at {ev['clips_per_s']:.0f} ten-second clips a second, {ev['realtime']:,} times real time and
{ev['clips_per_s'] / one['clips_per_s']:.1f} times one GPU's {one['clips_per_s']:.0f}. One GPU runs about as fast as
`Ensemble(path, bf16=True)` in batches of 64 ({eng['ensemble_bf16_b64_clips_per_s']:.0f} a second): a clip costs six
forward passes, {eng['tflop_per_clip']:.2f} TFLOP, so the GPU's compute is the limit, and what the engine adds is the
work spread over GPUs and concurrent requests batched together. An idle engine answers a clip in
{one['latency_ms_p50']:.0f} ms ({eager['latency_ms_p50']:.0f} ms without CUDA graphs). Over HTTP on {len(ev['gpus'])} GPUs
(the server in one process, clients posting 10 s WAV files), one client gets each answer in
{alone['latency_ms_p50']:.0f} ms; {some['concurrency']} clients at once get {some['req_per_s']:.0f} answers a second
({some['latency_ms_p50']:.0f} ms at the median), and {many['concurrency']} at once {many['req_per_s']:.0f} a second.

## What each member adds

{img['members.png']}

How much the ensemble loses without each member, on the same clips. The three released taggers
Melati-AudioSet-EAT-Large v4 was distilled from average {s['base3']:.4f}; with v3 and v4 that is {s['base3_v3_v4']:.4f},
with BEATs {s['base3_beats']:.4f}, and with all three {s['all']:.4f}. {misses} v3 and v4 are the same stage of Melati-AudioSet-EAT-Large trained twice, with and without a frame-level loss, so each
adds little once the other is in ({adds['melati-v3']:+.4f} and {adds['melati-v4']:+.4f}).{idle_txt}

## Members

| member | where it is loaded from | licence | parameters | mAP alone |
|---|---|---|---:|---:|
{members}

## Labels

{labels(res, res['clips'], named='`model.labels`')}

## Licence

`melati_ensemble.py` and this card are MIT. The member networks in `melati_ensemble.py` are written after their repos'
modeling code (EAT's, MIT; CED's, Apache-2.0), and its preprocessing and resampler after torchaudio's (BSD-2-Clause).
BEATs's weights and code are Microsoft's, MIT (`melati_beats/LICENSE`); the code is unilm's, unchanged but for its
imports of itself, made relative, and its torchaudio import, moved into the one method that uses it. The other members'
weights are fetched from their own repos under their own licences: Apache-2.0 (CED-base) and MIT (EAT-large, SSLAM,
Melati-AudioSet-EAT-Large). AudioSet's labels are CC BY 4.0 (Google); its audio is YouTube's and is not redistributed
here.

## Citation

```
@misc{{zolkepli2026melati,
  title  = {{Melati: Multilingual Expressive speech, Labelled for Affect, Tags and Interjections}},
  author = {{Husein Zolkepli}},
  year   = {{2026}},
  note   = {{Scicom (MSC) Berhad}}
}}
```

We thank Scicom (MSC) Berhad for the compute.
"""


def tree(api, repo, revision):
    """{path: an LFS file's sha256, else its git blob id} at a revision."""
    return {f.path: f.lfs.sha256 if f.lfs else f.blob_id
            for f in api.list_repo_tree(repo, recursive=True, expand=True, revision=revision) if hasattr(f, "blob_id")}


def update(api, a):
    """A new version of the public repo, in three steps run one after another (deploy/box_audioset_ensemble_update.sh),
    audioset/check_ensemble_fresh.py run against the staged commit between the first two:
      --stage          build, and upload the code to a staging branch made from main (a leftover one is replaced)
      --check-staging  the staged commit loaded from the Hub and scored (check()); the card written from that check and
                       from the new-environment one of the same commit, uploaded to the branch, and its Use snippet run
      --merge          the staged files uploaded to main, main confirmed to hold exactly them, the branch deleted"""
    assert api.repo_exists(a.repo) and not api.model_info(a.repo).private, f"{a.repo}: --push makes a new repo"
    if a.stage:
        build(a.out)
        if any(b.name == "staging" for b in api.list_repo_refs(a.repo).branches):
            print("[update] replacing a leftover staging branch", flush=True)
            api.delete_branch(a.repo, branch="staging")
        api.create_branch(a.repo, branch="staging")
        rev = api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", revision="staging",
                                commit_message=a.message, ignore_patterns=["README.md", "figures/*"]).oid
        print(f"STAGED {rev}", flush=True)
    elif a.check_staging:
        rev = next(b.target_commit for b in api.list_repo_refs(a.repo).branches if b.name == "staging")
        res = check(a.repo, rev, a.batch)
        OUT.write_text(json.dumps(res, indent=1))
        img = figures(a.out, a.repo, res)
        text = card(a.repo, res, img)
        assert "—" not in text, "no em dashes"
        (a.out / "README.md").write_text(text)
        api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", revision="staging",
                          allow_patterns=["README.md", "figures/*"],
                          commit_message=f"Card: {res['map']:.4f} mAP on AudioSet's evaluation set")
        snippet(a.repo, a.out / "README.md", revision="staging")
        print(f"CHECKED {rev}", flush=True)
    else:
        staged = tree(api, a.repo, "staging")
        api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", commit_message=a.message)
        main = tree(api, a.repo, "main")
        assert main == staged, sorted(p for p in set(main) | set(staged) if main.get(p) != staged.get(p))
        api.delete_branch(a.repo, branch="staging")
        api.add_collection_item(COLLECTION, item_id=a.repo, item_type="model", exists_ok=True)
        print(f"MERGED https://huggingface.co/{a.repo}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--message", default="A new version", help="the commit message of an update")
    ap.add_argument("--push", action="store_true", help="upload (private), check from the Hub, card, make public")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--card-only", action="store_true",
                    help="the card and figures again from the last check (results/audioset/ensemble_fp32.json), "
                         "uploaded beside the files that check scored, then the rest of --push")
    ap.add_argument("--stage", action="store_true", help="a new version of the public repo: build it and upload its "
                    "code to a staging branch made from main")
    ap.add_argument("--check-staging", action="store_true", help="check the staging branch from the Hub, and card it "
                    "there from that check and from audioset/check_ensemble_fresh.py's of the same commit")
    ap.add_argument("--merge", action="store_true", help="the staged files onto main, confirmed the same, and the "
                    "staging branch deleted")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    if a.stage or a.check_staging or a.merge:
        update(api, a)
        return
    if a.card_only:
        res = json.loads(OUT.read_text())
        assert res["repo"] == a.repo, (res["repo"], a.repo)
        files = {f.path: f.lfs.sha256 if f.lfs else f.blob_id
                 for f in api.list_repo_tree(a.repo, recursive=True, expand=True) if hasattr(f, "blob_id")}
        checked = {f.path: f.lfs.sha256 if f.lfs else f.blob_id
                   for f in api.list_repo_tree(a.repo, recursive=True, expand=True, revision=res["revision"])
                   if hasattr(f, "blob_id")}
        assert {p: h for p, h in files.items() if p in checked} == checked, "the repo's files moved since the check"
    else:
        build(a.out)
        print("files:", sorted(str(p.relative_to(a.out)) for p in a.out.rglob("*") if p.is_file()), flush=True)
        if not a.push:
            print("BUILT (--push uploads it, checks it from the Hub and publishes it)", flush=True)
            return
        assert not api.repo_exists(a.repo) or api.model_info(a.repo).private, \
            f"{a.repo} is public: a new version of it goes through --stage, --check-staging and --merge"
        api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
        rev = api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model",
                                commit_message="The ensemble: its code, BEATs, and each member's pinned revision").oid
        res = check(a.repo, rev, a.batch)
        OUT.write_text(json.dumps(res, indent=1))
    img = figures(a.out, a.repo, res)
    text = card(a.repo, res, img)
    assert "—" not in text, "no em dashes"
    (a.out / "README.md").write_text(text)
    api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", allow_patterns=["README.md", "figures/*"],
                      commit_message=f"Card: {res['map']:.4f} mAP on AudioSet's evaluation set")
    snippet(a.repo, a.out / "README.md")
    api.add_collection_item(COLLECTION, item_id=a.repo, item_type="model", exists_ok=True)
    api.update_repo_settings(a.repo, repo_type="model", private=False)
    print(f"pushed https://huggingface.co/{a.repo}", flush=True)
    print("RELEASE_DONE", flush=True)


if __name__ == "__main__":
    main()
