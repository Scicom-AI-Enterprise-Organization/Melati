#!/usr/bin/env python3
"""The card of Scicom-intl/Melati-Emotion-Heads: the five emotion heads Melati's pipeline averages with MERaLiON-SER
and emotion2vec+, written from the results the retrain left (deploy/box_emo_retrain.sh): what each head reads, how
it scores on clips no head trained on, the seven-model average and its thresholds, and every corpus the heads trained
on with that corpus's licence (emotion/release.py names the nine whose licence allows a derivative, and the rest).
RUNS ON THE BOX (emotion/release.py needs torch). deploy/box_emo_release_heads.sh runs it.

    python emotion/heads_card.py --results $MELATI_DATA/results/emotion --out <release folder>/README.md
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "emotion"))
from release import EXCLUDED, LANG_NAMES, TRAIN  # noqa: E402

REPO = "Scicom-intl/Melati-Emotion-Heads"
CODE = "https://github.com/Scicom-AI-Enterprise-Organization/Melati"
HEADS = [   # file, the encoder it reads
    ("whisper", "[openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)"),
    ("whisper_ft", "[Melati-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Whisper-Large-V3) at cdbb981c"),
    ("w2vbert", "[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)"),
    ("wavlm", "[microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)"),
    ("e2v", "[emotion2vec+ large](https://huggingface.co/emotion2vec/emotion2vec_plus_large)'s utterance embedding"),
]
OTHERS = {  # the corpora emotion/release.py's EXCLUDED names: corpus -> (name, licence)
    "crema_d": ("CREMA-D", "ODbL"), "ravdess": ("RAVDESS", "CC BY-NC-SA 4.0"), "cafe": ("CaFE", "CC BY-NC-SA 4.0"),
    "pavoque": ("PAVOQUE", "CC BY-NC-SA 4.0"), "meld": ("MELD", "GPL-3.0"),
    "nvtts_expresso": ("NonverbalTTS's labels on Expresso", "CC BY-NC-SA 4.0"),
    "turevdb": ("TurEV-DB", "CC BY-NC-ND 4.0"), "enterface": ("eNTERFACE", "no licence text"),
    "emodb": ("EmoDB", "no licence text"), "ased": ("ASED", "no licence text"), "shemo": ("ShEMO", "no licence text"),
    "urdu": ("URDU", "no licence text"),
}
LANGS = dict(LANG_NAMES, am="Amharic", de="German", fa="Persian", tr="Turkish", ur="Urdu")
LABELS = ("happy", "sad", "angry", "surprised", "fearful")


def card(res: Path) -> str:
    stats = json.loads((res / "manifest_stats.json").read_text())
    cmp = json.loads((res / "compare_v1.json").read_text())
    new, old = cmp["new"], cmp["old"]
    corpora = {c: v for c, v in stats["per_corpus"].items() if v["split"] == "train"}
    for name, lic in OTHERS.values():
        assert name.split(",")[0].split("'")[0] in EXCLUDED and lic in EXCLUDED, (name, lic)
    clean = {c: (n, lic) for c, n, lic, _ in TRAIN}
    assert set(corpora) == set(clean) | set(OTHERS), sorted(set(corpora) ^ (set(clean) | set(OTHERS)))
    rows = [(clean[c][0] if c in clean else f"{OTHERS[c][0]}, {LANGS[v['lang']]}", v["n"],
             (clean.get(c) or OTHERS[c])[1]) for c, v in corpora.items()]
    rows.sort(key=lambda r: (r[2] == "no licence text", r[0].lower()))
    assert sum(r[1] for r in rows) == stats["train"]["clips"]
    held = {c: v["n"] for c, v in stats["per_corpus"].items() if v["split"] == "test"}
    seven = new["average"]["seven"]
    head_rows = "\n".join(f"| `{f}.pt` | {enc} | {new['heads'][f]['held_out_all']:.3f} | "
                          f"{old['heads'][f]['held_out_all']:.3f} |" for f, enc in HEADS)
    cv = seven["thresholds_cv"]
    thr_rows = "\n".join(f"| {lab} | {seven['thresholds'][lab]:.2f} | {100 * cv[lab]['precision_at_prior']:.0f}% | "
                         f"{100 * cv[lab]['recall']:.0f}% |" for lab in LABELS)
    corpus_rows = "\n".join(f"| {n} | {k:,} | {lic} |" for n, k, lic in rows)
    langs = sorted(LANGS[v["lang"]] for v in corpora.values())
    langs = ", ".join(dict.fromkeys(langs))
    n, clips, n_held = len(rows), stats["train"]["clips"], sum(held.values())
    acc, pair = 100 * seven["held_out_acc"], new["average"]["meralion+e2v"]["held_out_uar"]
    return f"""---
license: other
license_name: melati-emotion-heads
pipeline_tag: audio-classification
tags:
- audio-classification
- speech-emotion-recognition
- emotion
- speech-data
---

# Melati emotion heads

The five emotion heads [Melati]({CODE})'s speech-data pipeline averages with MERaLiON-SER and emotion2vec+ to put
emotion scopes into podcast transcripts (`<happy>...</happy>`). Each is a small head on a frozen encoder: every encoder
layer mean-pooled over the clip's own frames, a learned softmax mix of the layers, LayerNorm, one 256-unit hidden layer
and seven outputs (neutral, happy, sad, angry, surprised, fearful, disgusted). They train on {n} emotional-speech
corpora; [Melati-Emotion-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3) is the
Whisper head trained on the nine whose licence explicitly allows a derivative.

| file | encoder | held-out UAR | the first heads' |
|---|---|---|---|
{head_rows}

Unweighted average recall on {n_held:,} clips no head trained on (nEMO, Polish, {held['nemo']:,}; NonverbalTTS's
VoxCeleb half, English, {held['nvtts_voxceleb']:,}). These heads were retrained on 2026-11-04 by
`deploy/box_emo_retrain.sh` after the box holding the first ones was wiped; the last column is the first heads', which
tagged the published [Melati dataset](https://huggingface.co/datasets/Scicom-intl/Melati). `results/` holds the counts
and scores behind every number here.

## In the pipeline

The seven models averaged (these five, MERaLiON-SER and emotion2vec+) score {seven['held_out_uar']:.3f} UAR and
{acc:.0f}% accuracy on the same clips, against {pair:.3f} for the
two public classifiers alone. A label is tagged only where its precision was measured: each threshold is the lowest
giving 75% precision at an assumed podcast rate, and chosen on one half of the held-out clips and measured on the
other it holds

| emotion | threshold | precision | recall |
|---|---|---|---|
{thr_rows}

Disgusted has no held-out positives and is never tagged. The thresholds live in Melati's
`melati/stage_export.py`.

## Use

With [Melati's code]({CODE}), into the folder the tag server reads (`deploy/box_melati_setup.sh` does this):

```bash
hf download {REPO} --include "*.pt" --local-dir "$MELATI_DATA/models/ser_heads"
```

```python
from melati.ser import HeadSER

head = HeadSER("whisper", heads_dir="ser_heads")   # loads openai/whisper-large-v3 at Melati's pinned revision
probs, _ = head(clips)                            # 16 kHz mono float32 clips, at most 30 s -> (clips, 7)
```

Each file holds the `encoder` it reads, its `labels`, the encoder's `layers` and `dim`, and `state`: the weights of
`melati.ser.head_module(layers, dim)`.

## Training data

{n} corpora, {clips:,} clips, in {langs}. Corpora are weighted equally in the loss, and classes
equally within each corpus.

| corpus | clips | licence |
|---|---|---|
{corpus_rows}

## Licence

For non-commercial research. The heads are trained on the corpora above, each under its own terms: several are
non-commercial or ShareAlike, TurEV-DB is NoDerivatives, and five come with no licence text. These weights do not lift
any of those terms; check them against your use.

## Limitations

- The training speech is mostly acted and none of it is Malay. On Malaysian podcasts, where Melati runs the heads,
  their labels are not yet measured against human judgements, and the podcast rates behind the thresholds are
  assumptions.

## Citation

```bibtex
@misc{{zolkepli2026melati,
  title  = {{Melati: Multilingual Expressive speech, Labelled for Affect, Tags and Interjections}},
  author = {{Husein Zolkepli}},
  year   = {{2026}},
  note   = {{Scicom (MSC) Berhad}}
}}
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="manifest_stats.json and compare_v1.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    text = card(args.results)
    assert "—" not in text, "no em dashes"
    args.out.write_text(text)
    print(f"{args.out}: {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
