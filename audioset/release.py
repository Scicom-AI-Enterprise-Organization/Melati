#!/usr/bin/env python3
"""Build, check and publish Melati's AudioSet tagger on Hugging Face. RUNS ON THE BOX, in .venv-sed.

One checkpoint: the architecture it started from, with the weights audioset/train.py distilled (the moving average at
its best evaluation) and AudioSet's label names in the config. EAT-large's (Chen et al., IJCAI 2024; its code, MIT,
copied from worstchan/EAT-large_epoch20_finetune_AS2M), or Dasheng-1.2B's (Dinkel et al., Interspeech 2024; its code,
Apache-2.0, copied from mispeech/dasheng-1.2B) with the head audioset/hub/modeling_melati_dasheng.py defines. Before anything is pushed the folder is loaded back with AutoModel
and scored on the evaluation set in fp32, and must give the mAP the results file says. The card's numbers and figures
come from results files: every model on the same 17,141 evaluation clips (audioset/evaluate.py --fp32), and the
training log.

    python audioset/release.py --ckpt $MELATI_DATA/models/audioset/clean_e10/best.pt \\
        --results $MELATI_DATA/results/audioset/student_clean_e10_fp32.json \\
        --out $MELATI_DATA/release/Melati-AudioSet-EAT-Large --repo Scicom-intl/Melati-AudioSet-EAT-Large [--push]

Every stage behind the checkpoint (its pool, settings and log) is read by following the checkpoint it started from. A
frame model (audioset/frames_train.py) is released by its clip half (audioset/clip_of_frames.py), and the card says how
its frame head compares with PretrainedSED's (audioset/frames_eval.py). A run of that script with the frame loss off
(--frame-weight 0) is a stage with no frame head; --with-frames names the same stage trained with one, and the card
sets the two side by side (v4 against v3, deploy/box_audioset_ship_v4.sh). The card lists every label with the model's
average precision on it (audioset/per_class.py), under the top-level categories of AudioSet's ontology
(deploy/box_audioset_card.sh fetches it). --card-only re-writes a released model's card and figures and leaves its
weights alone.
"""
import argparse
import json
import os
import re
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from data import ROOT, classes, load  # noqa: E402
from evaluate import mean_ap  # noqa: E402
from zoo import REPOS, EATLike  # noqa: E402

CODE = "https://github.com/Scicom-AI-Enterprise-Organization/Melati"
RESULTS = ROOT.parent / "results" / "audioset"
ONTOLOGY = ROOT / "ontology.json"         # github.com/audioset/ontology (Google, CC BY-SA 4.0)
NAMES = {"ced-base": "CED-base", "eat-large": "EAT-large", "eat-base": "EAT-base", "sslam": "SSLAM", "ast": "AST",
         "beats": "BEATs"}
CITE = {"ced-base": "Dinkel et al., ICASSP 2024", "eat-large": "Chen et al., IJCAI 2024",
        "eat-base": "Chen et al., IJCAI 2024", "sslam": "Alex et al., ICLR 2025", "ast": "Gong et al., Interspeech 2021",
        "beats": "Chen et al., ICML 2023"}
LICENCES = {"ced-base": "Apache-2.0", "eat-large": "MIT", "eat-base": "MIT", "sslam": "MIT", "beats": "MIT",
            "ast": "BSD-3-Clause"}
# teachers audioset/targets_add.py made from our own releases (deploy/box_audioset_v5.sh): a pool's name for each,
# and the results key its scores are filed under (the checkpoint's path on the box that scored it: a name, not a path
# to read)
SELF = {"v3": "melati:/mnt/data/melati/models/audioset/frames1/clip.pt",
        "v4": "melati:/mnt/data/melati/models/audioset/frames1_fw0/clip.pt"}
ENSEMBLE = ("Scicom-intl/Melati-AudioSet-Ensemble",        # the six's average itself (audioset/release_ensemble.py)
            {"ced-base", "eat-large", "sslam", "beats", "v3", "v4"})
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


EAT_USE = """```python
import torch, torchaudio, soundfile as sf
from transformers import AutoModel

model = AutoModel.from_pretrained("REPO_ID", trust_remote_code=True).eval()

wav, sr = sf.read("clip.wav", dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(1))
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
wav = wav - wav.mean()
fbank = torchaudio.compliance.kaldi.fbank(wav[None], htk_compat=True, sample_frequency=16000, use_energy=False,
                                          window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
fbank = torch.nn.functional.pad(fbank, (0, 0, 0, max(0, 1024 - len(fbank))))[:1024]     # 10.24 s
x = ((fbank + 4.268) / (4.569 * 2))[None, None]                                          # [1, 1, 1024, 128]

with torch.no_grad():
    probs = torch.sigmoid(model(x))[0]
for i in probs.topk(5).indices.tolist():
    print(model.config.id2label[i], round(probs[i].item(), 3))
```

The input is EAT's: 16 kHz mono, 128-bin Kaldi filterbanks every 10 ms, 1024 frames (a longer clip in 10 s windows)."""

DASHENG_USE = """```python
import torch, torchaudio, soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel

fe = AutoFeatureExtractor.from_pretrained("REPO_ID", trust_remote_code=True)
model = AutoModel.from_pretrained("REPO_ID", trust_remote_code=True).eval()

wav, sr = sf.read("clip.wav", dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(1))
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)

with torch.no_grad():
    probs = torch.sigmoid(model(**fe(wav, sampling_rate=16000)).logits)[0]
for i in probs.topk(5).indices.tolist():
    print(model.config.id2label[i], round(probs[i].item(), 3))
```

The input is Dasheng's own: 16 kHz mono, a 64-bin mel spectrogram every 10 ms from its feature extractor. A clip
longer than 10 s is cut inside the model into 10 s windows, and the head reads all of their tokens. The feature
extractor sets its dB floor 120 dB under the loudest value in the whole batch it is given, so a clip with digital
silence in it (padding) can score slightly differently beside louder clips; one clip at a time is the reference."""

BASES = {
    "eat-large": {"title": "Melati-AudioSet-EAT-Large", "license": "mit", "headless": False,
                  "what": "EAT-large (309M parameters)",
                  "start": "From EAT-large fine-tuned on AudioSet-2M ({start:.4f} on these clips in bf16)",
                  "use": EAT_USE,
                  "licence": "MIT, as EAT's code and weights it starts from."},
    "dasheng-1.2b": {"title": "Melati-AudioSet-Dasheng-1.2B", "license": "apache-2.0", "headless": True,
                     "what": "Dasheng-1.2B (1.2B parameters; Dinkel et al., Interspeech 2024), a self-supervised "
                             "encoder with no AudioSet head of its own,",
                     "start": "From Dasheng-1.2B's self-supervised weights and a new head ({start:.4f} on these clips "
                              "before any training)",
                     "use": DASHENG_USE,
                     "licence": "Apache-2.0, as Dasheng's code and weights it starts from (Xiaomi); the head and its "
                                "code (`modeling_melati_dasheng.py`) are released under the same licence."},
}


SOURCES = (("strong_train", "AudioSet's strongly labelled training clips ({n:,}, every event timed by hand, from "
                           "[CLAPv2/audioset_strong](https://huggingface.co/datasets/CLAPv2/audioset_strong))"),
           ("bal_train", "its balanced set ({n:,})"), ("unbal_train", "{n:,} clips of its unbalanced set"))


def label(k, key, stages, names=None):
    """A results key as the card names it: this model, an earlier stage of it, or a released model."""
    if k == key:
        return "this model"
    if names and k in names:
        return names[k]
    if k.startswith("melati:"):
        for n, stg in enumerate(stages, 1):
            if str(stg["ckpt"]) == k.split(":", 1)[1]:
                return f"this model after stage {n}"
        return "an earlier Melati model"
    return NAMES.get(k, k)


def clip_map(r):
    """A log record's clip mAP: the moving average's (audioset/train.py), or the frame model's clip head's."""
    return r.get("map_ema", r.get("map", r.get("clip_map")))


def lineage(ckpt):
    """The training stages behind a checkpoint, earliest first: each stage's pool, arguments, log and best mAP, found
    by following the checkpoint it started from (audioset/train.py --init melati:<path>). A frame model's stage also
    carries its frame head's comparison with PretrainedSED, when audioset/frames_eval.py has scored it."""
    out, path = [], Path(ckpt)
    while True:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        rec = [json.loads(line) for line in (path.parent / "log.jsonl").read_text().splitlines()]
        args = next(r["args"] for r in rec if "args" in r)
        pool = json.loads((ROOT / "cache" / f"pool_{args['pool']}_meta.json").read_text())
        stage = {"ckpt": path, "args": args, "log": rec, "pool": pool, "map": ck["map"],
                 "frames": float(args.get("frame_weight", 0)) > 0}   # --frame-weight 0 trains no frame head
        fe = RESULTS / f"frames_eval_{path.parent.name}.json"
        if stage["frames"] and fe.exists() and ck.get("frames") in json.loads(fe.read_text()):
            r = json.loads(fe.read_text())
            stage["frame_eval"] = {"ours": r[ck["frames"]]["shared"], "pretrainedsed": r["pretrainedsed"]["shared"],
                                   "classes": r["classes_shared"], "clips": r["clips"]}
        out.append(stage)
        frm = str(ck.get("from", ck["init"]))
        if not frm.startswith("melati:"):
            return out[::-1], ck["init"]
        path = Path(frm.split(":", 1)[1])


SHORT = {"strong_train": "the strongly labelled set", "bal_train": "the balanced set",
         "unbal_train": "{n:,} clips of the unbalanced set"}


def pool_words(pool, short=False):
    """A pool's sources in words, AudioSet named first; short, by the names pool_words gave them in full."""
    parts = [(SHORT[k] if short else txt).format(n=pool["by_source"][k]) for k, txt in SOURCES if k in pool["by_source"]]
    return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]


def lr_words(x):
    return f"{float(x):g}".replace("e-0", "e-")


def average(res, teachers):
    """The results file's mAP for the average of a pool's teachers (evaluate.py --averages)."""
    key = " + ".join(sorted(SELF.get(t, t) for t in teachers))
    assert key in res["averages"], f"the results hold no average of {teachers}: evaluate.py --averages"
    return res["averages"][key]


def joined(teachers):
    """Teachers in words, the released ones by name and then our own: 'BEATs and its own v3 and v4'."""
    theirs = [NAMES[t] for t in teachers if t not in SELF]
    own = [t for t in teachers if t in SELF]
    parts = ([", ".join(theirs[:-1]) + " and " + theirs[-1] if len(theirs) > 1 else theirs[0]] if theirs else []) \
        + ([f"its own {' and '.join(own)}"] if own else [])
    return " and ".join(parts)


def build(ckpt, out):
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    src = Path(snapshot_download(REPOS[ck["init"]], allow_patterns=["*.json", "*.py"]))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.suffix in (".py", ".json"):
            shutil.copy(f, out / f.name)
    cfg = json.loads((out / "config.json").read_text())
    names = [n for _, _, n in classes()]
    cfg.update(id2label={str(i): n for i, n in enumerate(names)}, label2id={n: i for i, n in enumerate(names)})
    if ck["init"].startswith("dasheng"):                    # Dasheng's code unchanged, the head in a file of its own
        shutil.copy(HERE / "hub" / "modeling_melati_dasheng.py", out / "modeling_melati_dasheng.py")
        cls = "modeling_melati_dasheng.DashengAudioSetTagger"
        cfg.update(architectures=["DashengAudioSetTagger"],
                   auto_map={"AutoConfig": "configuration_dasheng.DashengConfig", "AutoModel": cls,
                             "AutoModelForAudioClassification": cls})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file({k: v.contiguous().float() if v.is_floating_point() else v.contiguous() for k, v in ck["state"].items()},
              str(out / "model.safetensors"), metadata={"format": "pt"})   # transformers refuses a file without it
    return ck


class HubDasheng:
    """A released Dasheng tagger as its card says to run it: its own feature extractor, then the model, in fp32."""

    def __init__(self, repo, revision=None):
        from transformers import AutoFeatureExtractor, AutoModel
        self.fe = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True, revision=revision)
        self.m = AutoModel.from_pretrained(repo, trust_remote_code=True, revision=revision).eval().cuda()

    @torch.no_grad()
    def __call__(self, xs):
        x = self.fe([np.asarray(v, np.float32) for v in xs], sampling_rate=16000)["input_values"].cuda()
        return torch.sigmoid(self.m(x).logits.float()).cpu().numpy()


def check(repo, expect, base, revision=None):
    """The uploaded repo, loaded from the Hub as a user would, scored in fp32 on the evaluation set. (From a local
    folder transformers 4.46 cannot load it: EAT's code imports model_core.py two levels down, and the module cache
    is read before it is filled.)"""
    import zoo
    zoo.BF16 = False
    m = HubDasheng(repo, revision) if base.startswith("dasheng") else EATLike(repo, revision=revision)
    W, _, _, Y = load("eval")
    P = np.concatenate([m([w.astype(np.float32) for w in W[i:i + 64]]) for i in range(0, len(W), 64)])
    got = mean_ap(P, Y)
    print(f"[check] released folder: mAP {got:.4f} (results file {expect:.4f})", flush=True)
    assert abs(got - expect) < 1e-3, (got, expect)
    return got


def snippet(repo, readme, revision=None):
    """The card's Use snippet run as a user would, on the CPU, on one evaluation clip (audioset/check_card_use.py)."""
    cmd = [sys.executable, str(HERE / "check_card_use.py"), "--repo", repo, "--readme", str(readme)]
    r = subprocess.run(cmd + (["--revision", revision] if revision else []), capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    print(r.stdout[-1500:], flush=True)
    assert r.returncode == 0 and "SNIPPET_OK" in r.stdout, r.stderr[-2000:]


def figures(out, repo, res, key, stages, base, names=None, twin=None):
    import matplotlib.pyplot as plt
    import melati.plotstyle as st
    (out / "figures").mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(out / "figures" / name, dpi=st.CARD_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
    # 1. every model on the same evaluation clips: a dot each, since the gaps are a fraction of a point
    ms = sorted(res["models"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    st.frame(ax, st.better("mAP on AudioSet's evaluation set", lower=False), None, axis="x")
    for i, (k, v) in enumerate(ms):
        ours = k == key
        ax.plot([0.45, v], [i, i], color=st.RULE, lw=0.8, zorder=1)
        ax.scatter(v, i, s=46 if ours else 32, color=st.S1 if ours else st.MUTED, zorder=3)
        ax.text(v + 0.0012, i, f"{v:.4f}", va="center", fontsize=7, color=st.TEXT, fontweight="bold" if ours else None)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([label(k, key, stages, names) for k, _ in ms], fontsize=7.6)
    ax.set_xlim(0.45, max(v for _, v in ms) + 0.007)
    save(fig, "map.png")
    # 2. training, stage after stage: the moving average's mAP at each evaluation, against the best released model
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    st.frame(ax, "training step", st.better("mAP (bf16)", lower=False))
    at, ys = 0, []
    for k, stg in enumerate(stages):
        start = at
        pts = [(at + r["step"], clip_map(r)) for r in stg["log"]   # a new head's start is no point on this scale
               if r.get("step", -1) >= 0 and clip_map(r) and (k == 0 and not BASES[base]["headless"] or r["step"] > 0)]
        ax.plot([x for x, _ in pts], [v for _, v in pts], color=(st.S1, st.S2, st.S5, st.S3)[k % 4], lw=1.6,
                marker="o", ms=3, zorder=3, label=f"stage {k + 1}: {stg['pool']['clips']:,} clips"
                f"{', with a frame head' if stg['frames'] else ''}"
                f"{', ' + str(len(stg['pool']['teachers'])) + ' teachers' if stg['pool']['teachers'] != stages[0]['pool']['teachers'] else ''}"
                if len(stages) > 1 else
                "this model (moving average)")
        if k:
            ax.axvline(at, color=st.RULE, lw=0.8, ls=":", zorder=1)
        ys += [v for _, v in pts]
        at = pts[-1][0]
    if twin:                                                   # the last stage trained with a frame head beside it
        pts = [(start + r["step"], clip_map(r)) for r in twin["log"] if r.get("step", -1) > 0 and clip_map(r)]
        ax.plot([x for x, _ in pts], [v for _, v in pts], color=st.MUTED, lw=1.2, ls="--", marker="o", ms=2.5,
                zorder=2, label=f"stage {len(stages)} with a frame head: {(names or {}).get(twin['key'], 'earlier run')}")
        ys += [v for _, v in pts]
    released = [v for k, v in res["models"].items() if not k.startswith("melati:")]
    best = max(released)
    ax.axhline(best, color=st.S4, lw=1.0, ls="--", zorder=2, label=f"best released model ({best:.4f}, fp32)")
    ax.set_ylim(min(ys + [best]) - 0.002, max(ys + [best]) + 0.003)
    st.legend_below(ax, ncol=2, pad=0.3)
    save(fig, "training.png")
    return {n: st.card_img(out / "figures" / n, alt, st.hf_url(repo, f"figures/{n}"))
            for n, alt in (("map.png", "mAP on AudioSet's evaluation set"), ("training.png", "Training"))}


def groups():
    """AudioSet's classes under the ontology's top-level categories, in its own order: [(category, [class index])],
    and how many classes the ontology reaches from more than one category (each is listed under the first)."""
    nodes = json.loads(ONTOLOGY.read_text())
    by_id = {n["id"]: n for n in nodes}
    child = {c for n in nodes for c in n["child_ids"]}
    index = {mid: i for i, mid, _ in classes()}

    def under(top):
        found, todo = set(), [top]
        while todo:
            m = todo.pop()
            if m in index:
                found.add(index[m])
            todo += by_id[m]["child_ids"]
        return found
    reach = [(n["name"], under(n["id"])) for n in nodes if n["id"] not in child]
    out, placed = [], set()
    for name, found in reach:
        out.append((name, sorted(found - placed)))
        placed |= found
    assert len(placed) == len(index), "every class sits under a top-level category"
    return out, sum(sum(i in f for _, f in reach) > 1 for i in index.values())


def labels(per, clips, named="`model.config.id2label`"):
    """The card's list of every label: a folded table for each of the ontology's top-level categories, each class
    with the model's average precision on it and how many evaluation clips carry it (audioset/per_class.py); named is
    what holds the label names in the card's Use snippet."""
    cls = per["classes"]
    assert not any("|" in c["name"] for c in cls)
    grouped, shared = groups()
    parts = []
    for name, idx in grouped:
        mean = np.mean([cls[i]["ap"] for i in idx if cls[i]["ap"] is not None])
        rows = "\n".join(f"| {i} | {cls[i]['name']} | {'n/a' if cls[i]['ap'] is None else format(cls[i]['ap'], '.3f')} "
                         f"| {cls[i]['positives']:,} |" for i in idx)
        parts.append(f"<details>\n<summary><b>{name}</b>: {len(idx)} labels, mean AP {mean:.3f}</summary>\n\n"
                     f"| index | label | AP | clips |\n|---:|---|---:|---:|\n{rows}\n\n</details>")
    missing = sum(c["ap"] is None for c in cls)
    return (f"All {len(cls)} of AudioSet's classes, as {named} names them (the index is the output's "
            f"column), under the {len(grouped)} top-level categories of "
            f"[AudioSet's ontology](https://github.com/audioset/ontology) (Google, CC BY-SA 4.0)"
            f"{f'; the {shared} classes it places under two categories are listed under the first' if shared else ''}. "
            f"AP is this model's average precision on the class over the {clips:,} evaluation clips in fp32, and clips "
            f"is how many of them carry the label; the mAP above is the mean of the AP column"
            f"{f', the {missing} classes no evaluation clip carries left out' if missing else ''}.\n\n"
            + "\n\n".join(parts))


def card(repo, res, key, stages, img, base, per, names=None, twin=None):
    b = BASES[base]
    ms = sorted(res["models"].items(), key=lambda kv: -kv[1])
    ours = res["models"][key]
    others = {k: v for k, v in res["models"].items() if not k.startswith("melati:")}
    best_k = max(others, key=others.get)
    named = [(n, res["models"][k]) for k, n in (names or {}).items() if k in res["models"]]
    short = [(m.group(1) if (m := re.search(r"\((v\d+)\)", n)) else n, v) for n, v in named]
    beaten = (f"the best, {NAMES[best_k]}, {others[best_k]:.4f}" if not named else
              f"the best of the others is {NAMES[best_k]}, {others[best_k]:.4f}), our own earlier "
              f"release{'s' if len(named) > 1 else ''} included ({', '.join(f'{n} {v:.4f}' for n, v in short)}")
    rows = "\n".join(f"| {'**this model**' if k == key else label(k, key, stages, names)} | {'**' if k == key else ''}{v:.4f}"
                     f"{'**' if k == key else ''} | {'' if k.startswith('melati:') else CITE.get(k, '')} |" for k, v in ms)
    pool = max((s["pool"] for s in stages), key=lambda p: p["clips"])          # the pools nest: the widest one
    start = next(r["map"] for r in stages[0]["log"] if r.get("step") == 0)
    fr = next((n for n, s in enumerate(stages, 1) if s["frames"]), None)
    mixed = [n for n, s in enumerate(stages, 1) if float(s["args"].get("mixup", 0)) > 0]
    hours = pool.get("hours")
    first = stages[0]["pool"]["teachers"]
    teachers = " + ".join(NAMES[t] for t in first)
    avg = average(res, first)
    moved = [n for n, s in enumerate(stages, 1) if s["pool"]["teachers"] != first]
    later = ""
    if moved:                                                  # a later stage learned from more teachers
        last = stages[moved[0] - 1]["pool"]["teachers"]
        assert all(s["pool"]["teachers"] == last for s in stages[moved[0] - 1:]), "one change of teachers"
        link = (f", released as [Melati-AudioSet-Ensemble](https://huggingface.co/{ENSEMBLE[0]})"
                if set(last) == ENSEMBLE[1] else "")
        later = (f" and, {'in its last stage' if moved == [len(stages)] else 'from stage ' + str(moved[0])}, of "
                 f"those {NUM[len(first)]} with {joined([t for t in last if t not in first])} "
                 f"({average(res, last):.4f} averaged{link})")
    ceiling = average(res, stages[-1]["pool"]["teachers"])
    every = list(dict.fromkeys(t for s in stages for t in s["pool"]["teachers"]))
    lic = {}
    for t in every:
        if t not in SELF:
            lic.setdefault(LICENCES[t], []).append(NAMES[t])
    own = [t for t in every if t in SELF]
    taught = ("The teachers it learned from are released under "
              + " and ".join(f"{k} ({', '.join(ns)})" for k, ns in lic.items())
              + (f"; {NUM[len(own)]} of them were its own earlier versions" if own else "") + ".")
    a0 = stages[-1]["args"]
    steps = "\n".join(
        f"{n}. {b['start'].format(start=start) if n == 1 else 'From stage ' + str(n - 1)}, "
        f"{'with a second head beside the first that places each sound in time, every 160 ms, ' if stg['frames'] else ''}"
        f"{float(stg['args']['epochs']):g} epochs on {pool_words(stg['pool'], short=True)} ({stg['pool']['clips']:,} clips, "
        f"{stg['pool']['hours']:,.0f} hours), "
        f"{'with ' + joined([t for t in stg['pool']['teachers'] if t not in first]) + ' added to the teachers, ' if stg['pool']['teachers'] != first else ''}"
        f"learning rate {lr_words(stg['args']['lr'])}"
        f"{' (the new head ' + format(float(stg['args']['head_lr_mult']), 'g') + ' times that)' if n == 1 and b['headless'] else ''}: {stg['map']:.4f} "
        f"{'where the frame head was best' if stg['frames'] else 'at its best'} (bf16)."
        for n, stg in enumerate(stages, 1))
    prev = f"melati:{stages[fr - 2]['ckpt']}" if fr and fr > 1 else None
    lift = (f"Stage {fr}'s frame head learns the hand-timed events of the {stages[fr - 1]['pool']['with_timed_events']:,} "
            f"strongly labelled clips (binary cross-entropy every 160 ms). The stage raised the clip head as well, "
            f"{res['models'][prev]:.4f} after stage {fr - 1} and {ours:.4f} after it on these clips in fp32; which part "
            f"of it did so (the timed events, a second learning-rate cycle, no mixup) is not yet separated. "
            if prev in res["models"] else "")
    if twin:
        n, before = len(stages), f"melati:{stages[-2]['ckpt']}" if len(stages) > 1 else None
        lift = (f"The {(names or {}).get(twin['key'], 'earlier run')} trained stage {n} with a second head beside the "
                f"first that learns the hand-timed events of the {twin['pool']['with_timed_events']:,} strongly "
                f"labelled clips (binary cross-entropy every 160 ms): {res['models'][twin['key']]:.4f} on these clips "
                f"in fp32. Trained again with that loss off, the stage reaches {ours:.4f}, so the timed events did not "
                f"lift the clip head, and this release leaves them out"
                + (f"; its lift over stage {n - 1} ({res['models'][before]:.4f}) comes from the rest of the stage, "
                   f"another learning-rate cycle on {pool_words(stages[-1]['pool'], short=True)} without mixup. "
                   if before in res["models"] else ". "))
    if not mixed:
        mixing = ""
    elif len(stages) == 1:
        mixing = "; half the batches were mixed (mixup)"
    elif len(mixed) == len(stages):
        mixing = "; every stage also mixed half its batches (mixup)"
    elif len(mixed) == 1:
        mixing = f"; stage {mixed[0]} also mixed half its batches (mixup)"
    else:
        mixing = f"; stages {', '.join(map(str, mixed[:-1]))} and {mixed[-1]} also mixed half their batches (mixup)"
    every = (f"{'Every stage' if len(stages) > 1 else 'Training'}: SpecAugment, warm-up and cosine decay, bf16 with "
             "gradient checkpointing, and an exponential "
             f"moving average of the weights ({float(a0['ema']):g}) whose best evaluation is kept{mixing}.")
    fe = stages[fr - 1].get("frame_eval") if fr else (twin or {}).get("frame_eval")
    which = (f"The frame head trained beside it in stage {fr} is not part of this release" if fr else
             f"The frame head trained beside the {(names or {}).get((twin or {}).get('key'), 'earlier run')} was not "
             f"released either")
    clip_level = (f"- Clip-level: a probability for each class over the whole input. {which}: on AudioSet's "
                  f"{fe['clips']:,} strongly labelled test clips it places sounds in time less well than PretrainedSED "
                  f"(frame mAP {fe['ours']:.4f} against {fe['pretrainedsed']:.4f} on the {fe['classes']} classes both "
                  f"have), which is what Melati's own pipeline uses for that." if fe else
                  "- Clip-level: a probability for each class over the whole input, not where in the clip each "
                  "sound is.")
    return f"""---
license: {b['license']}
library_name: transformers
pipeline_tag: audio-classification
base_model: {REPOS[base]}
tags:
- audio
- audio-classification
- audioset
- sound-event-detection
- distillation
---

# {b['title']}

An AudioSet tagger: one model that names which of AudioSet's [527 sounds](#labels) are in a clip, scoring **{ours:.4f} mAP** on
AudioSet's evaluation set, ahead of every released single model we scored on the same clips
({beaten}). It is {b['what']} distilled from the average of {NUM[len(first)]} released
taggers ({teachers}; {avg:.4f} averaged){later} on {pool['clips']:,} AudioSet clips ({hours:,.0f} hours), none of them from
its evaluation set{'; its last stage also trained a frame-level head beside it on the hand-timed events of AudioSet' + chr(39) + 's strongly labelled clips' if fr else ''}. Part of [Melati]({CODE}), Scicom's speech-data pipeline.

{img['map.png']}

| model | mAP | |
|---|---|---|
{rows}

All on the same {res['clips']:,} clips: AudioSet's evaluation set as
[agkphysics/AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) holds it today (YouTube removes clips over
time, so every paper's copy differs and published numbers are not comparable across papers), each model in fp32 with
its own preprocessing, mAP the mean over the 527 classes of their average precision.

## Use

{b['use'].replace("REPO_ID", repo)}

## Labels

{labels(per, res['clips'])}

## How it was trained

{img['training.png']}

The {'clip head' if fr else 'student'} is trained to the average of the teachers' probabilities (binary cross-entropy against soft targets;
AudioSet's own {'clip ' if fr else ''}labels are not used) on {pool_words(pool)}, any clip of the evaluation set left out, in
{len(stages)} stage{'s' if len(stages) > 1 else ''}:

{steps}

{lift}{every} The code is [audioset/]({CODE}/tree/main/audioset) in Melati.

## Limitations

- One evaluation set: AudioSet's own. On speech-heavy material (podcasts) it tags what AudioSet tags, which is not a
  measure of how well it times or places an event inside a long recording.
{clip_level}
- Its teachers' average ({ceiling:.4f}) bounds what distilling from them can reach; it is {ceiling - ours:.4f} under it.
- The training clips are {pool['clips']:,} of AudioSet's 2 million; more of its unbalanced set is the obvious lever.

## Licence

{b['licence']} {taught} AudioSet's labels are CC BY 4.0 (Google); its audio is YouTube's and is not redistributed
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--name", action="append", default=[], metavar="KEY=NAME",
                    help="what the card calls another model in the results, e.g. melati:<ckpt>=Melati-AudioSet-EAT-Large")
    ap.add_argument("--with-frames", metavar="KEY",
                    help="the results key of this model's last stage trained again with a frame head beside it "
                         "(audioset/frames_train.py), which the card sets beside this one")
    ap.add_argument("--card-only", action="store_true",
                    help="re-write a released model's card and figures, its weights left as they are")
    ap.add_argument("--message", help="the commit message (default: the release's)")
    a = ap.parse_args()
    names = dict(n.split("=", 1) for n in a.name)
    res = json.loads(a.results.read_text())
    key = f"melati:{a.ckpt}"
    assert key in res["models"], f"{key} is not in {a.results}"
    for k, n in names.items():                              # a release beats every released model, ours included
        assert res["models"].get(k, 0) < res["models"][key], f"{n} ({res['models'].get(k)}) is not beaten"
    pc = RESULTS / f"per_class_{a.ckpt.parent.name}.json"
    assert pc.exists(), f"no {pc}: run audioset/per_class.py --ckpt {a.ckpt} --results {a.results}"
    per = json.loads(pc.read_text())
    assert per["model"] == key and per["clips"] == res["clips"] and abs(per["map"] - res["models"][key]) < 1e-4, pc
    stages, base = lineage(a.ckpt)
    twin = None
    if a.with_frames:
        assert a.with_frames in res["models"], f"{a.with_frames} is not in {a.results}"
        twin = dict(lineage(Path(a.with_frames.split(":", 1)[1]))[0][-1], key=a.with_frames)
        assert twin["frames"] and not stages[-1]["frames"], "--with-frames: a frame model beside one without"
        assert twin["args"]["init"] == stages[-1]["args"]["init"], "--with-frames: not the same stage"
    if a.card_only:
        a.out.mkdir(parents=True, exist_ok=True)
    else:
        build(a.ckpt, a.out)
    img = figures(a.out, a.repo, res, key, stages, base, names, twin)
    text = card(a.repo, res, key, stages, img, base, per, names, twin)
    assert not re.search("—", text), "no em dashes"
    (a.out / "README.md").write_text(text)
    print("files:", sorted(p.name for p in a.out.iterdir()), flush=True)
    if a.push and a.card_only:
        # the card waits on a staging branch until its own Use snippet runs from there
        from huggingface_hub import HfApi
        api = HfApi()
        assert api.repo_exists(a.repo), f"{a.repo} has no release to re-card"
        msg, only = a.message or "Model card", ["README.md", "figures/*"]
        api.create_branch(a.repo, branch="staging", exist_ok=True)
        api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", commit_message=msg,
                          revision="staging", allow_patterns=only)
        snippet(a.repo, a.out / "README.md", revision="staging")
        api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", commit_message=msg,
                          allow_patterns=only)
        api.delete_branch(a.repo, branch="staging")
        print(f"pushed https://huggingface.co/{a.repo}", flush=True)
    elif a.push:
        # a new repo stays private, and a new version of a public one waits on a staging branch, until the upload,
        # loaded from the Hub as a user would, gives the results file's mAP
        from huggingface_hub import HfApi
        api = HfApi()
        msg = a.message or f"Melati's AudioSet tagger: {res['models'][key]:.4f} mAP on AudioSet's evaluation set"
        if not api.repo_exists(a.repo):
            api.create_repo(a.repo, repo_type="model", private=True)
            api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", commit_message=msg)
            check(a.repo, res["models"][key], base)
            snippet(a.repo, a.out / "README.md")
            api.update_repo_settings(a.repo, repo_type="model", private=False)
        else:
            api.create_branch(a.repo, branch="staging", exist_ok=True)
            api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", commit_message=msg,
                              revision="staging")
            check(a.repo, res["models"][key], base, revision="staging")
            snippet(a.repo, a.out / "README.md", revision="staging")
            api.upload_folder(repo_id=a.repo, folder_path=str(a.out), repo_type="model", commit_message=msg)
            api.delete_branch(a.repo, branch="staging")
        print(f"pushed https://huggingface.co/{a.repo}", flush=True)
    print("RELEASE_DONE", flush=True)


if __name__ == "__main__":
    main()
