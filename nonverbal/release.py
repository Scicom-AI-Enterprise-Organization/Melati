#!/usr/bin/env python3
"""Build, check and publish Melati's non-verbal model on Hugging Face. RUNS ON THE BOX.

One checkpoint (model.safetensors): the encoder of openai/whisper-large-v3 in float16 and two heads in float32,
loaded by AutoModel with trust_remote_code (melati_hf/modeling_melati_head.py):

  gap head      what a sound cut out of a gap between two aligned words is (nonverbal/train.py); it tags breaths
  frame head    every 0.1 s of a segment, laughter and coughs, over speech too (nonverbal/frames.py); it tags laughs
                and coughs

Beside it: the Whisper feature extractor's config, the tagging helper (nonverbal/hf/melati_nonverbal.py), the
licence, and a card whose numbers and figures come from results files. Before anything is pushed, the merged model
must reproduce the pipeline's own models (melati.ser.HeadNV, melati.nv_frames.FrameNV).

Licence: PodcastFillers' annotations are licensed for non-commercial research only and its licence must travel with
derivatives, so the weights are released for non-commercial research only with that licence enclosed. They saw only
PodcastFillers episodes whose audio is CC BY 3.0, AMI, COUGHVID and Coswara (CC BY 4.0). NonverbalTTS's labels are CC
BY-NC-SA 4.0 (its README; its metadata tag says apache-2.0) and ShareAlike cannot hold beside PodcastFillers' licence,
so it only scores the model. The release before this one (rel2) did train on it, and is replaced.

    python nonverbal/release.py --head rel4 --frames 3 --pf-run rel4f --out $MELATI_DATA/release/Melati-NonVerbal \\
        --repo Scicom-intl/Melati-NonVerbal-Whisper-Large-V3 [--push]
"""
import argparse, datetime, json, os, re, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from melati import nv_gaps as NG  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

RULE = '{"floor_db": 2}'
R = Path(f"{MELATI_DATA}/results")
MD = Path(f"{MELATI_DATA}/models")
PREV = "rel2"                                    # the release this one replaces
CODE = "https://github.com/Scicom-AI-Enterprise-Organization/Melati"
RELEASE_DATE = None


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def pct(v, nd=0):
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def adobe_licence(pf_readme: Path):
    """The PodcastFillers metadata licence, as its README carries it (it must travel with derivatives)."""
    t = pf_readme.read_text(encoding="utf-8", errors="replace")
    a = t.index("## License for PodcastFillers Dataset metadata")
    b = t.index("## License for PodcastFillers Dataset audio files")
    return t[a:b].strip()


def licence_text(pf_readme):
    return f"""Melati non-verbal model: licence

The weights in this repository may be used, reproduced, modified and redistributed for NON-COMMERCIAL RESEARCH
PURPOSES ONLY. Non-commercial research purposes include academic research and teaching only; they do not include
commercial licensing or distribution, development of commercial products, or any other activity that results in
commercial gain. Any redistribution of the weights or of works derived from them must carry this licence, including
the PodcastFillers licence below.

Why: the model was trained on PodcastFillers (Zhu, Caceres and Salamon, Interspeech 2022), whose annotations Adobe
licenses for non-commercial research only and whose licence must be given to recipients of derivative works.

Training data and attribution (every source used to train the weights; the heads were trained on frozen features
of this material and do not contain it):
- PodcastFillers annotations: Adobe Inc., licence reproduced below. PodcastFillers audio: only episodes released
  under CC BY 3.0 (100 of 199; https://creativecommons.org/licenses/by/3.0/), by their SoundCloud authors as listed
  in PodcastFillers' LICENSE_AUDIO.csv. Episodes under CC BY-SA 3.0 or CC BY-ND 3.0 were not used for training.
- The AMI Meeting Corpus (Carletta et al. 2005), audio and manual annotations, CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/): https://groups.inf.ed.ac.uk/ami/corpus/
- COUGHVID (Orlandic, Teijeiro and Atienza, Scientific Data 2021; CC BY 4.0): https://zenodo.org/records/7024894
- Coswara (Bhattacharya et al., Scientific Data 2023; CC BY 4.0): https://github.com/iiscleap/Coswara-Data
- Changes: clips and 30 s windows were cut out of these recordings, resampled to 16 kHz mono and, for clips, padded
  with silence before the frozen encoder read them; nothing of them is redistributed here.
- The encoder is openai/whisper-large-v3 (MIT), downloaded from its own repository, not redistributed here.

Not used for training: NonverbalTTS (deepvk; its annotations are CC BY-NC-SA 4.0) only scores the model.
VocalSound (CC BY-SA 4.0) was not used.

The software in melati_nonverbal.py is part of Melati ({CODE}).

---------------------------------------------------------------------------------------------------------------
{adobe_licence(pf_readme)}
"""


def lab_p(d, c):
    """Precision counting only tags where PodcastFillers has a label (it labelled only what its detector proposed)."""
    m = d["scopes"]["inside"][c]
    if not m.get("pred"):
        return None
    hit = round(m["precision"] * m["pred"])
    wrong = sum(v for k, v in d["unmatched_predictions_overlap"].get(c, {}).items() if k != "(no label)")
    return hit / max(hit + wrong, 1)


class Numbers:
    """Every number the card and its figures show, read from results files."""

    def __init__(self, head, frames, pf_run):
        self.thr_gap = {c: v["threshold"] for c, v in J(R / f"nonverbal/gap_eval_{head}pf.json")[RULE]["thresholds"].items()
                        if v and c == "breath"}
        fe = J(R / f"nonverbal/frames{frames}_eval_pf.json")
        self.thr_frames = {k: v for k, v in (("laugh", (fe.get("podcastfillers") or {}).get("threshold")),
                                             ("cough", fe["ami"]["cough"]["threshold"])) if v}
        self.fe = fe
        self.pf, self.pf_prev, self.pf_omni = (J(R / f"fillers/podcastfillers_nv{v}.json") if v != "rel" else
                                               J(R / "fillers/podcastfillers_rel.json") for v in (pf_run, PREV, "rel"))
        self.audit = J(R / f"fillers/audit_nv_tags_{pf_run}.json") or {}
        self.ami_prev = J(R / f"nonverbal/ami_eval_{PREV}.json")["test"]["nonverbaltts_thresholds"]
        self.nv_prev = J(R / f"nonverbal/gap_eval_{PREV}.json")[RULE]["test"]
        self.curve = J(R / f"nonverbal/gap_curve_{head}.json")["curves"]["test"]
        self.fnv = J(R / f"nonverbal/frames{frames}_nvtts.json")["test"]
        self.held = J(R / f"nonverbal/head_whisper_{head}.json")["test"]

    def nv_breath(self):                         # the gap head on NonverbalTTS test at the breath threshold that ships
        r = self.curve["breath"][str(self.thr_gap["breath"])]
        return r["tag_recall"], r["tag_precision"]

    def nv_frames(self, cls):                   # the frame head on NonverbalTTS test at the threshold that ships
        t = self.thr_frames.get(cls)
        r = (self.fnv.get(cls) or {}).get(str(t)) if t else None
        return (r["recall"], r["precision"]) if r else (None, None)

    def ami_frames(self, cls):                  # the frame head on AMI test at the threshold that ships
        a = self.fe["ami"][cls]
        return a["test_at_pf_threshold"] if cls == "laugh" and "test_at_pf_threshold" in a else a["test"]

    @staticmethod
    def pf_class(d, c):
        m = d["scopes"]["inside"][c]
        return (m["recall"] if m.get("pred") else None), lab_p(d, c)


def figures(out: Path, N: Numbers):
    """The card's charts, one per file, in the preprint's style (melati/plotstyle.py)."""
    import melati.plotstyle as st
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    pct_axis = matplotlib.ticker.PercentFormatter(1.0, decimals=0)

    def save(fig, name):
        (out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "figures" / name, dpi=st.CARD_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
    # 1. PodcastFillers' test episodes through the pipeline
    groups = [("laughter", 0, "laughs\nfound"), ("laughter", 1, "laughs precise\nwhere labelled"),
              ("breath", 0, "breaths\nfound"), ("breath", 1, "breaths precise\nwhere labelled")]
    runs = [(N.pf_omni, st.RULE, "Qwen3-Omni-30B"), (N.pf_prev, st.S4, "previous release"), (N.pf, st.S1, "this model")]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    st.frame(ax, None, st.better("inside what Melati exports", lower=False), axis="y")
    w = 0.27
    for r, (d, c, lab) in enumerate(runs):
        for i, (cls, what, _) in enumerate(groups):
            v = N.pf_class(d, cls)[what]
            x = i + (r - 1) * w
            ax.bar(x, v or 0.0, w * 0.92, color=c, zorder=3, label=lab if i == 0 else None)
            ax.text(x, (v or 0.0) + 0.015, f"{100 * v:.0f}%" if v is not None else "none", ha="center", va="bottom",
                    fontsize=6.3, color=st.TEXT)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[2] for g in groups], fontsize=7.4)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(pct_axis)
    st.legend_below(ax, ncol=3, pad=0.26)
    save(fig, "podcastfillers.png")
    # 2. meetings and interviews: found, precision in the bar; the previous release beside
    rp = lambda r: {"recall": r[0], "precision": r[1]}  # noqa: E731
    pv = lambda c: {"recall": N.nv_prev[c]["tag_recall"], "precision": N.nv_prev[c]["tag_precision"]}  # noqa: E731
    groups = [("AMI laughs", N.ami_frames("laugh"), N.ami_prev["laugh"]),
              ("AMI coughs", N.ami_frames("cough"), N.ami_prev["cough"]),
              ("NonverbalTTS\nlaughs", rp(N.nv_frames("laugh")), pv("laugh")),
              ("NonverbalTTS\nbreaths", rp(N.nv_breath()), pv("breath")),
              ("NonverbalTTS\ncoughs", rp(N.nv_frames("cough")), pv("cough"))]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    st.frame(ax, None, st.better("found", lower=False), axis="y")
    for off, which, c, lab in ((-0.2, 2, st.S4, "previous release"), (0.2, 1, st.S1, "this model")):
        for i, g in enumerate(groups):
            r = g[which] or {}
            v = r.get("recall") or 0.0
            ax.bar(i + off, v, 0.38, color=c, zorder=3, label=lab if i == 0 else None)
            short = r.get("precision") is not None and 0 < v <= 0.12      # no room inside: precision on top
            top = (f"{100 * v:.0f}%\n{100 * r['precision']:.0f}%\nprecise" if short else
                   f"{100 * v:.0f}%" if r.get("precision") is not None else "none")
            ax.text(i + off, v + 0.015, top, ha="center", va="bottom", fontsize=5.8 if short else 6.5, color=st.TEXT,
                    linespacing=1.0)
            if r.get("precision") is not None and v > 0.12:
                ax.text(i + off, v / 2, f"{100 * r['precision']:.0f}%\nprecise", ha="center", va="center", fontsize=5.6,
                        color="white", linespacing=1.1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(pct_axis)
    st.legend_below(ax, ncol=2, pad=0.22)
    save(fig, "meetings_interviews.png")
    # 3. laughs on PodcastFillers at every threshold of the frame head (nonverbal/sweep_frames.py): the other two are
    # single points, and the threshold, not the model, decides where this one sits beside them
    sw = J(R / "nonverbal/frames_sweep_pf.json")
    if sw:
        pts = sorted((float(t), v) for t, v in sw["laugh"].items())
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        st.frame(ax, st.better("laughs found", lower=False), st.better("precise where labelled", lower=False))
        ax.plot([v["recall"] for _, v in pts], [v["precision_labelled"] for _, v in pts], color=st.S1, lw=1.6,
                marker="o", ms=3.2, zorder=3, label="this model, threshold by threshold")
        where = {N.thr_frames.get("laugh"): (5, -11), N.fe["ami"]["laugh"]["threshold"]: (7, 3), 0.99: (-9, 7)}
        for t, v in pts:
            if t in where:                       # placed clear of the line and of the other two models' points
                ax.annotate(f"{t:g}", (v["recall"], v["precision_labelled"]), textcoords="offset points",
                            xytext=where[t], fontsize=6.6, color=st.S1)
        for name, c in (("previous release", st.S4), ("Qwen3-Omni", st.MUTED)):
            o = sw["others"][name]
            ax.scatter(o["recall"], o["precision_labelled"], s=30, color=c, zorder=4, label=name, edgecolor="white",
                       linewidth=0.6)
        ax.set_xlim(0.0, 0.8)
        ax.set_ylim(0.7, 1.0)
        ax.xaxis.set_major_formatter(pct_axis)
        ax.yaxis.set_major_formatter(pct_axis)
        st.legend_below(ax, ncol=2, pad=0.26)
        save(fig, "podcastfillers_laughs.png")


def img(out, repo, name, alt):
    """The card's <img> for figures/<name>, at the scale every Melati card shows its figures at (melati/plotstyle.py)."""
    import melati.plotstyle as st
    return st.card_img(out / "figures" / name, alt, st.hf_url(repo, f"figures/{name}"))


def card(cfg, repo, N: Numbers, out):
    fh = cfg["frame_head"]
    pf_rows = []
    for c, name in (("laughter", "laughs"), ("breath", "breaths")):
        a, b, q = (N.pf_class(d, c) for d in (N.pf, N.pf_prev, N.pf_omni))
        pf_rows.append(f"| {name} | {pct(a[0])} / {pct(a[1])} | {pct(b[0])} / {pct(b[1])} | "
                       + (f"{pct(q[0])} / {pct(q[1])}" if q[0] is not None else "not tagged") + " |")
    al, ac = N.ami_frames("laugh"), N.ami_frames("cough")
    nvl, nvb, nvc = N.nv_frames("laugh"), N.nv_breath(), N.nv_frames("cough")
    p = N.nv_prev
    other_rows = [
        f"| AMI test meetings, laughs | {pct(al['recall'])} at {pct(al['precision'])} "
        f"({pct(al['recall_in_speech'])} of those over speech) | "
        f"{pct(N.ami_prev['laugh']['recall'])} at {pct(N.ami_prev['laugh']['precision'])} |",
        f"| AMI test meetings, coughs | {pct(ac['recall'])} at {pct(ac['precision'])} | "
        f"{pct(N.ami_prev['cough']['recall'])} |",
        f"| NonverbalTTS test, laughs | {pct(nvl[0])} at {pct(nvl[1])} | "
        f"{pct(p['laugh']['tag_recall'])} at {pct(p['laugh']['tag_precision'])} |",
        f"| NonverbalTTS test, breaths | {pct(nvb[0])} at {pct(nvb[1])} | "
        f"{pct(p['breath']['tag_recall'])} at {pct(p['breath']['tag_precision'])} |",
        "| NonverbalTTS test, coughs | " + (f"{pct(nvc[0])} at {pct(nvc[1])}" if nvc[1] is not None else "none tagged")
        + f" | {pct(p['cough']['tag_recall'])} at {pct(p['cough']['tag_precision'])} |"]
    names = {"podcastfillers": "PodcastFillers' test episodes", "coughvid": "COUGHVID", "coswara": "Coswara",
             "ami_cands": "AMI's test meetings"}
    f1 = "; ".join(f"{names[k]}: " + ", ".join(f"{c} {m['f1']:.2f}" for c, m in v["per_class"].items() if m["n"] >= 10)
                   for k, v in N.held.items() if k in names)
    sw, curve = J(R / "nonverbal/frames_sweep_pf.json"), ""
    t_ami = N.fe["ami"]["laugh"]["threshold"]
    if sw and str(t_ami) in sw["laugh"]:
        at, pv = sw["laugh"][str(t_ami)], sw["others"]["previous release"]
        curve = f"""**Laughs at every threshold.** This model's laugh precision above is where its threshold sits, not what it
can do: at each threshold of the frame head, on the same episodes through the same pipeline, it passes the previous
release's point. At {t_ami:g}, the threshold AMI's dev meetings choose, it finds {pct(at['recall'])} of the laughs at
{pct(at['precision_labelled'])} precision, against the previous release's {pct(pv['recall'])} at
{pct(pv['precision_labelled'])}; the {N.thr_frames.get('laugh'):g} it ships with trades that for more laughs. The
[Melati dataset](https://huggingface.co/datasets/Scicom-intl/Melati) tags laughs at {t_ami:g}.

{img(out, repo, "podcastfillers_laughs.png", "Laughs on PodcastFillers at every threshold")}

"""
    lu, lc = N.audit.get("laughter/unlabelled") or {}, N.audit.get("laughter/control") or {}
    bu, bc = N.audit.get("breath/unlabelled") or {}, N.audit.get("breath/control") or {}
    return f"""---
license: other
license_name: melati-nonverbal-research
license_link: LICENSE
base_model: openai/whisper-large-v3
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio-classification
- non-verbal
- laughter
- breath
- cough
- speech-data
---

# Melati non-verbal model (Whisper-large-v3)

Breaths, laughs and coughs in speech, with their times. Melati's speech-data pipeline uses it to put Drama-3-style
point tags (`[breathes]`, `[laughs]`, `[coughs]`) into podcast transcripts, at the time each sound happened. Code,
training and evaluation: [Melati on GitHub]({CODE}).

One checkpoint: the encoder of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) (float16)
and two small heads on its frozen features (float32):

- **gap head**: what a sound cut out of a gap between two aligned words is ({", ".join(cfg['labels'])}). Every
  encoder layer mean-pooled over the clip's own frames, a learned mix of the layers, LayerNorm, one 256-unit hidden
  layer. It tags **breaths**.
- **frame head**: every 0.1 s of a segment, an independent probability of {", ".join(fh['classes'])}. Eight encoder
  layers averaged to 10 frames a second, a learned mix of them, and four residual dilated convolutions over time
  (about 6 s of context). It tags **laughs** and **coughs**, including the ones over speech, which no gap holds.

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
probs = model(f, num_frames=frames).logits.softmax(-1)      # gap head, over model.config.labels
per_frame = model.frames(f)                                 # frame head: (1, 300, 4), 10 a second over the 30 s
```

To tag a transcribed segment the way Melati does, `melati_nonverbal.py` (in this repository) finds the gap
candidates, reads the frames, and applies the thresholds:

```python
from huggingface_hub import hf_hub_download
import importlib.util
spec = importlib.util.spec_from_file_location("melati_nonverbal", hf_hub_download(repo, "melati_nonverbal.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

nv = m.NonVerbal.from_pretrained(repo)
events = nv.tag(audio_16k, words)   # words: [{{"s": 0.31, "e": 0.62}}, ...] from a forced aligner
```

A gap candidate is sound in a gap between words (and before the first, after the last) above both the clip's noise
floor + {cfg['candidate_rule']['floor_db']:g} dB and its speech level {cfg['candidate_rule']['rel_db']:g} dB for at
least {int(cfg['candidate_rule']['min_active'] * 1000)} ms, cut out with {cfg['candidate_rule']['pad_s']:g} s of
silence on either side; it is a breath where that is its most probable non-verbal class and clears
{N.thr_gap['breath']}. A laugh or a cough is a run of frames over its threshold (laugh {N.thr_frames.get('laugh')},
cough {N.thr_frames.get('cough')}), joined across up to two quiet frames and at least 0.2 s long. Each threshold is
the lowest giving 80% precision where it was chosen: breaths and laughs on PodcastFillers' validation episodes, coughs
on AMI's dev meetings (PodcastFillers labels none).

## Evaluation

**PodcastFillers test episodes** (English podcasts, 20 episodes from shows nothing here trained on), through the
whole Melati pipeline, inside the regions it exports. PodcastFillers labelled only what its own detector proposed,
so precision is given where it has a label (a tag where it has none is unscored, not wrong). Found / precise:

{img(out, repo, "podcastfillers.png", "PodcastFillers through the pipeline")}

| class | this model | previous release | Qwen3-Omni-30B |
|---|---|---|---|
{chr(10).join(pf_rows)}

The tags PodcastFillers has no label for were played to AudioSet's own classifier (AST), which no part of Melati
trained: unlabelled laugh tags score {lu.get('mean_Laughter', 0):.3f} on its Laughter class against
{lc.get('mean_Laughter', 0):.3f} for hand-labelled laughs, and {pct(bu.get('breath_like'))} of unlabelled breath tags
sound to it breath-like (gasp, sigh, breathing) against {pct(bc.get('breath_like'))} of hand-labelled breaths.

{curve}**Meetings and interviews**, at the same thresholds. AMI's test meetings (16 meetings of four people nothing here
trained on; a tag is right where the transcribers marked a sound of its class nearby) and NonverbalTTS's test split
(359 clips of interviews and expressive read speech, word-boundary tags; right within one word):

{img(out, repo, "meetings_interviews.png", "Meetings and interviews: found, and how many tags are right")}

| test | this model: found at precision | previous release |
|---|---|---|
{chr(10).join(other_rows)}

On held-out clips of its training corpora, the gap head's F1 at the most probable class (classes with at least ten
of them): {f1}.

## Training data

- [PodcastFillers](https://podcastfillers.github.io/) training episodes whose audio is CC BY 3.0: for the gap head,
  its hand-timed events and the gap candidates the pipeline's own rule finds there, labelled by the event each
  overlaps; for the frame head, every 30 s window holding a labelled event, its classes known inside the labelled
  regions only.
- [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) training meetings (CC BY 4.0), the headset mix: every word and every
  marked vocal sound of a whole conversation. For the gap head, its gap candidates (unmarked ones as `other`); for the
  frame head, every frame.
- Coughs from [COUGHVID](https://zenodo.org/records/7024894) (those its authors' detector is confident about) and
  [Coswara](https://github.com/iiscleap/Coswara-Data) (those its listeners rated good or excellent), both CC BY 4.0.

NonverbalTTS only scores the model (its labels are CC BY-NC-SA 4.0). VocalSound and PodcastFillers' CC BY-SA and
CC BY-ND episodes were left out so the weights can carry one licence.

## Changes

Until {RELEASE_DATE} this repository held a gap head alone that also trained on NonverbalTTS's candidates, and its
licence file called NonverbalTTS Apache-2.0, which is what its metadata says; its README licenses the annotations CC
BY-NC-SA 4.0, and ShareAlike cannot hold beside PodcastFillers' licence in one set of weights. This version trains on
AMI's meetings in their place, takes its laughs and coughs from the new frame head, and chooses its thresholds on
PodcastFillers' validation episodes instead of NonverbalTTS: on NonverbalTTS it is now tested out of domain.

## Limitations

- Trained and measured on English: podcasts, meetings, interviews and crowdsourced coughs. On Malaysian podcasts,
  where Melati runs it, its tags are not yet measured against human labels.
- The thresholds were chosen on podcasts and meetings; on NonverbalTTS's interviews the table above is what they give.
- Coughs are the weak class: the frame head finds them in meetings, over speech too, but at the threshold chosen on
  AMI's dev meetings it tags none of NonverbalTTS's and none in Melati's 200-podcast pilot (the previous release
  tagged 0.2 an hour there). No licence-compatible corpus labels coughs in podcasts to choose a threshold on.
- Sighs, sniffs, sneezes and throat clearing have outputs but no thresholds: nothing measured them on test.

## Licence

Non-commercial research only, with the PodcastFillers licence; see [LICENSE](LICENSE).

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


def smoke(out: Path, head_dir: Path, frames_dir: Path, n=64):
    """The merged model, loaded as the public loads it, against the pipeline's own models."""
    sys.path.insert(0, str(out))
    sys.path.insert(0, str(ROOT / "nonverbal"))
    from gap_rule import load
    from melati_nonverbal import NonVerbal
    from melati.nv_frames import FrameNV
    from melati.ser import HeadNV
    clips, segs = [], []
    for cl, x in load(Path(f"{MELATI_DATA}/nv/nvtts_align.json"), "test"):
        cs = NG.candidates(x, cl["words"])
        clips += [NG.clip_of(x, c) for c in cs]
        if cs and len(segs) < 8:
            segs.append((x, cl["words"]))
        if len(clips) >= n:
            break
    nv = NonVerbal(out)
    a = nv.classify(clips[:n])
    b = HeadNV("whisper", heads_dir=str(head_dir))(clips[:n])[0]
    d = float(np.abs(a - b).max())
    same = float((a.argmax(1) == b.argmax(1)).mean())
    fa = [nv.frames(x) for x, _ in segs]
    fb = FrameNV("whisper", models_dir=str(frames_dir))([x for x, _ in segs])[0]
    df = max(float(np.abs(p - q).max()) for p, q in zip(fa, fb))
    tags = [nv.tag(x, w) for x, w in segs]
    print(f"[smoke] {len(clips[:n])} candidates: max |dp| {d:.5f}, same top class {same:.3f}; frames on {len(segs)} "
          f"clips: max |dp| {df:.5f}; tag(): {[len(t) for t in tags]} events, by "
          f"{sorted({e['by'] for t in tags for e in t})}", flush=True)
    assert d < 5e-3 and same == 1.0 and df < 5e-3, (d, same, df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", default="rel4", help="the gap head: models/nv_heads_<head>, gap_eval_<head>pf.json")
    ap.add_argument("--frames", default="3", help="the frame head: models/nv_frames<frames>, frames<frames>_eval_pf.json")
    ap.add_argument("--pf-run", default="rel4f", help="the pipeline's PodcastFillers test run with both models")
    ap.add_argument("--date", default=None, help="the day the previous release was replaced (default today)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pf-readme", type=Path,
                    default=Path(f"{MELATI_DATA}/bench/podcastfillers/PodcastFillers/README.txt"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-only", action="store_true", help="upload the release already built in --out")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the card's figures into --out/figures and stop (scripts/push_card_figures.py uploads them)")
    ap.add_argument("--card-only", action="store_true",
                    help="redraw the figures and rewrite README.md in a release already built in --out, weights untouched "
                         "(scripts/push_card_figures.py --readme uploads them); pass the release's --date")
    args = ap.parse_args()
    if args.push_only:
        return push(args)
    global RELEASE_DATE
    RELEASE_DATE = args.date or datetime.date.today().isoformat()
    N = Numbers(args.head, args.frames, args.pf_run)
    if args.figures_only:
        return figures(args.out, N)
    if args.card_only:
        figures(args.out, N)
        (args.out / "README.md").write_text(card(json.loads((args.out / "config.json").read_text()), args.repo, N,
                                                 args.out))
        assert not re.search("—", (args.out / "README.md").read_text()), "no em dashes"
        return print(f"card -> {args.out / 'README.md'}")
    head = MD / f"nv_heads_{args.head}" / "whisper.pt"
    frames = MD / f"nv_frames{args.frames}" / "whisper.pt"
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    from melati_hf.merge import build_merged
    build_merged(head, args.out, frames=frames, frame_thresholds=N.thr_frames, thresholds=N.thr_gap, task="nonverbal",
                 candidate_rule={"rel_db": NG.REL_DB, "floor_db": NG.FLOOR_DB, "min_active": NG.MIN_ACTIVE,
                                 "dilate": 0.03, "max_len": 2.5, "pad_s": NG.PAD_S})
    cfg = json.loads((args.out / "config.json").read_text())
    shutil.copy(ROOT / "nonverbal/hf/melati_nonverbal.py", args.out / "melati_nonverbal.py")
    (args.out / "LICENSE").write_text(licence_text(args.pf_readme))
    figures(args.out, N)
    (args.out / "README.md").write_text(card(cfg, args.repo, N, args.out))
    assert not re.search("—", (args.out / "README.md").read_text()), "no em dashes"
    smoke(args.out, head.parent, frames.parent)
    shutil.rmtree(args.out / "__pycache__", ignore_errors=True)        # the smoke test imported the module from here
    print("files:", sorted(str(p.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file()))
    if args.push:
        push(args)
    print("RELEASE_DONE", flush=True)


def push(args):
    from huggingface_hub import HfApi
    assert (args.out / "model.safetensors").exists() and (args.out / "README.md").exists(), "build it first"
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(repo_id=args.repo, folder_path=str(args.out), repo_type="model", delete_patterns=["figures/*"],
                      ignore_patterns=["__pycache__/*", "*.pyc"],
                      commit_message="Licence-clean: no NonverbalTTS in training; AMI, and a frame head for laughs and coughs")
    print(f"pushed https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
