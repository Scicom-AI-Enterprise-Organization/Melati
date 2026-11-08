# Melati

**M**ultilingual **E**xpressive speech, **L**abelled for **A**ffect, **T**ags and **I**nterjections.

An in-the-wild speech-data pipeline in the line of [Emilia-Pipe](https://arxiv.org/abs/2501.15907), and the dataset it
builds. Every segment carries word times, speaker turns, quality scores and tags for **emotion**, **non-verbal sounds**
and **fillers**, written in [Fish Audio Drama 3](https://huggingface.co/spaces/fishaudio/Drama-3)'s tag syntax:

```
<happy>hello [coughs] sorry, um, <fast>I just got the results</fast></happy> [laughs]
```

The tags are a rendering of time-stamped events (word times, event onsets and offsets, emotion spans), which are kept
in every row, so a different target format never needs re-annotation.

## The dataset

[Scicom-intl/Melati](https://huggingface.co/datasets/Scicom-intl/Melati): every recording of
[malaysia-ai/malaysian-podcast-youtube](https://huggingface.co/datasets/malaysia-ai/malaysian-podcast-youtube)
(19,092 Malaysian podcasts, about 2,200 hours) through the pipeline below, one row per segment with its 24 kHz FLAC
inline, CC BY-NC 4.0 as its source. The first release holds 6,912 of the recordings: 355,477 segments, 591 hours of
Malay and English (`results/dataset/stats.json`). It is published in shards of 64 recordings, and its card counts what
is there.

```bash
bash deploy/box_full.sh                  # every recording through front to fill, on serve/cluster.json
bash deploy/box_full_publish_loop.sh     # pack what has finished, check it, upload it; again every 6 h, and once at the end
python scripts/full_progress.py ...      # how far it is; scripts/cluster_load.py: where the GPU time goes
```

## Models

Every model the pipeline runs is public except two sets of weights you train yourself (see Getting started): the five
emotion heads and the frame head behind the inline sound tags. Each card states its licence in full.

Released with Melati:

| model | what it does | stage | licence |
|---|---|---|---|
| [Melati-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Whisper-Large-V3) | transcription that writes nothing over audio with no speech in it | asr, fill | CC BY-NC 4.0 |
| [Melati-NonVerbal-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-NonVerbal-Whisper-Large-V3) | breaths, laughs and coughs, with their times | tag | non-commercial research |
| [Melati-Emotion-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3) | seven emotions: the Whisper emotion head, retrained on the corpora whose licence allows a release | not run: the pipeline's heads train on more corpora | non-commercial research |
| [Melati-AudioSet-EAT-Large](https://huggingface.co/Scicom-intl/Melati-AudioSet-EAT-Large) | which of AudioSet's 527 sounds a clip holds | sounds | MIT |
| [Melati-AudioSet-Ensemble](https://huggingface.co/Scicom-intl/Melati-AudioSet-Ensemble) | six AudioSet taggers averaged | sounds, with `serve/cluster_sounds_ensemble.json` | MIT |

Scicom's models it runs:

| model | what it does | stage | licence |
|---|---|---|---|
| [Who-Spoke-When-Over-the-Phone-Whisper-Large-V3-Merged](https://huggingface.co/Scicom-intl/Who-Spoke-When-Over-the-Phone-Whisper-Large-V3-Merged) | speaker turns | diar | CC BY-NC 4.0 |
| [Where-Words-Begin-and-End-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Where-Words-Begin-and-End-Whisper-Large-V3) | where every word starts and ends | align | CC BY-NC 4.0 |
| [SOTA-MOS](https://huggingface.co/Scicom-intl/SOTA-MOS) | mean opinion score, 1 to 5 | quality | MIT |

And from elsewhere: [DBS-TIGER-DnR](https://huggingface.co/JusperLee/DBS-TIGER-DnR) separates speech from music and
effects where [PANNs Cnn14](https://zenodo.org/records/3987831) hears background (front), [Silero
VAD](https://github.com/snakers4/silero-vad) finds the pauses segments are cut in (seg),
[DNSMOS](https://github.com/microsoft/DNS-Challenge) scores quality beside SOTA-MOS, and
[MERaLiON-SER](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) and
[emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_large) are two of the seven emotion models. The
other five, Melati's emotion heads, read [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3),
Melati-Whisper-Large-V3, [w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0),
[WavLM-large](https://huggingface.co/microsoft/wavlm-large) and emotion2vec+'s features; the non-verbal heads read
whisper-large-v3.

## The pipeline

Every GPU model sits behind its own server, and the pipeline is a client of them (`melati/pipeline.py`, driven by
`melati/run.py`). Emilia-Pipe runs the whole pipeline once per GPU worker, so every worker holds every model and each
model only ever sees one file's segments at a time; here each model is scaled on its own, by adding replicas or
raising a vLLM service's tensor-parallel size in `serve/cluster.json`, and dynamic batching fills every GPU call with
pieces of many recordings.

| stage | model (server) | Emilia-Pipe used |
|---|---|---|
| front: decode, one gain per recording, separation where needed | PANNs events + DBS-TIGER (`serve/front_server.py`) | UVR-MDX-Net on everything, peak normalisation |
| diar: who spoke when | Who-Spoke-When-Over-the-Phone (vLLM) | pyannote 3.1 |
| seg: single-speaker segments, cut only in pauses | Silero VAD (CPU) | Silero VAD on turns over 20 s, hard 30 s cuts |
| asr: transcription | [Melati-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Whisper-Large-V3), a merge of two whisper-large-v3 fine-tunes (vLLM) | Whisper medium (WhisperX) |
| align: word times | Where-Words-Begin-and-End (`serve/align_server.py`) | none |
| quality | SOTA-MOS at 24 kHz (its server) and DNSMOS (`serve/tag_server.py`), on mix and separated speech | DNSMOS at 16 kHz |
| tag: emotion | seven models averaged: our heads on five frozen encoders (`melati/ser.py`), MERaLiON-SER and emotion2vec+ (`serve/tag_server.py`) | none |
| tag: non-verbal | breaths: sound in the gaps between aligned words (`melati/nv_gaps.py`), labelled by our gap head on Whisper-large-v3 (`nv_whisper`); laughs and coughs, over speech too: our frame head on the same encoder (`melati/nv_frames.py`, `nvf_whisper`) | none |
| fill: fillers in voiced gaps | the ASR server, on each gap alone | none |
| export | FLAC 24 kHz + one manifest row per segment, nothing filtered | MP3 at 32 kbit/s, filtered |

```bash
python serve/launch.py up --config serve/cluster.json
python -m melati.run --sources sources.jsonl --work work/ --out export/
python -m melati.run --sources sources.jsonl --work work/ --out export/ --assemble
```

Every stage resumes by output existence and writes atomically, and every threshold lives in one export config, so a
subset is a query rather than a re-run.

## Getting started

Melati runs on Linux with NVIDIA GPUs. The scripts in `deploy/` were written for one shared machine with eight GPUs,
and each cluster config in `serve/` names the GPUs every server uses (`replicas`): edit those for your own.

**Settings.** `deploy/env.sh` reads them from the environment or from `$MELATI_ROOT/.env`, which also holds `HF_TOKEN`
and is never committed:

| variable | holds | default |
|---|---|---|
| `MELATI_ROOT` | the Python environments, third-party checkouts and `.env` | `/root/melati` |
| `MELATI_DATA` | data, models, results and the Hugging Face cache | `$MELATI_ROOT/data` |
| `MELATI_LOGS` | the logs of long runs | `$MELATI_ROOT/logs` |
| `PIP_MIRROR` | a PyPI mirror for the setup scripts | none |

Keep `MELATI_DATA` outside the clone. The Python code reads `MELATI_DATA` and `MELATI_ROOT` as well, and the cluster
configs write their paths as `${MELATI_DATA}/...`, which `serve/launch.py` fills in.

**Environments and public weights.** Each setup script builds one environment under `$MELATI_ROOT` and downloads what
it serves:

```bash
export MELATI_ROOT=/opt/melati MELATI_DATA=/data/melati
bash deploy/box_melati_setup.sh    # .venv-melati (stages, servers, tagging) and .venv-vllm (the ASR)
bash deploy/box_diar_setup.sh      # .venv-diar (the diarization model's vLLM)
bash deploy/box_align_setup.sh     # .venv-align (word times)
bash deploy/box_sed_setup.sh       # .venv-sed (the sound tagger)
$MELATI_ROOT/.venv-melati/bin/python -m pytest tests/
```

**Released weights, in the pipeline's files.** The non-verbal gap and frame heads (`nv_whisper`, `nvf_whisper`) are
released together as [Melati-NonVerbal-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-NonVerbal-Whisper-Large-V3)
for `AutoModel`, while the pipeline reads each from its own file under `$MELATI_DATA/models`. `nonverbal/from_release.py`
writes those files from the release (`deploy/box_melati_setup.sh` runs it; with `--check` it compares them with the
release on a GPU). `deploy/box_nv_rel4.sh` and `deploy/box_nv_frames3.sh` train them again.

**What you train yourself.** Two sets of weights the pipeline reads are not on the Hub:

- the five emotion heads (`melati/ser.py`), since several of their 21 training corpora forbid passing a derivative on:
  `deploy/box_emo_setup.sh`, then `deploy/box_emo_retrain.sh`, which downloads the corpora (about 28 GB). Without
  them the tag server serves only MERaLiON-SER and emotion2vec+, and the export tags their average at the thresholds
  measured for that pair, which hold the seven models' precision for happy, angry and surprised only;
- the frame head behind the inline sound tags (`serve/sounds_server.py`, `FRAMES_CKPT`), trained beside the AudioSet
  tagger's v3 by `audioset/frames_train.py` (`deploy/box_audioset_frames.sh`), which needs AudioSet itself. Without
  it the sounds stage still gives every segment its sounds (the `sounds` column) and writes no inline sound tags.

These are the scripts that made the released models, run in order on one machine; they have not been run again from a
clean start. With every model in place, list the recordings (`python -m melati.sources --dir <audio> --list
sources.jsonl`) and run the commands above.

## Checks

`ruff check .` and `python -m pytest` need no GPU and no model weights, only
`pip install ruff==0.14.3 pytest numpy soundfile pyloudnorm pyarrow`; their settings are in `pyproject.toml`.

## The ASR model

`asr/` fine-tunes whisper-large-v3 with the recipe of
[How to Reduce Whisper Hallucination](https://arxiv.org/abs/2609.32560) (empty targets on voice-free audio, synthesised
hallucination phrases as positives) and adds Common Voice 22 as replay, because every fine-tune in that paper lost
multilingual accuracy. The released model, [Scicom-intl/Melati-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Whisper-Large-V3),
averages the weight changes of two of them (`asr/train/merge_models.py`) and is scored as vLLM serves it, with no
decoding constraint. See `asr/`, `deploy/box_asr_*.sh` and `deploy/box_hf_release.sh`.

## The tag classifiers

Every tag class is kept only where its precision was measured on labelled clips no model trained on, at a threshold
chosen on separate clips; the thresholds live in `melati/stage_export.py`.

- **Emotion** (`emotion/`): a small head (a learned mix of an encoder's layers, mean-pooled, one hidden layer) on
  each of Whisper-large-v3, Melati's Whisper fine-tune, w2v-BERT 2.0, WavLM-large and emotion2vec+'s features,
  trained on every openly downloadable EmoBox corpus plus CAMEO, Expresso and MELD; the seven models (these five, and
  MERaLiON-SER and emotion2vec+) are averaged. `emotion/fetch_corpora.py`, `build_manifest.py`, `extract.py`,
  `probe.py`, `ensemble.py`; MER2023 (gated) through `emotion/fetch_mer2023.py` and `deploy/box_emo_mer2023.sh`.
  The Whisper head retrained on only the nine corpora whose licence allows a release (six languages) is released for
  non-commercial research as
  [Scicom-intl/Melati-Emotion-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3)
  (`deploy/box_emo_release_head.sh`, `emotion/release.py`; helper `emotion/hf/melati_emotion.py`).
- **Non-verbal** (`nonverbal/`): two heads on the frozen Whisper-large-v3 encoder.
  - The **gap head** (the emotion heads' design, eleven outputs: speech, filler, breath, laugh, cough, sigh, sniff,
    throat clearing, sneeze, noise, other) labels the candidates `melati/nv_gaps.py` cuts out of each word gap, and
    tags **breaths**. It trains on PodcastFillers' CC BY episodes (events, and the gap candidates the pipeline's own
    rule finds, `nonverbal/candidates.py`), AMI's meetings (`nonverbal/ami.py`, CC BY 4.0: candidates labelled by the
    transcribers' marks, unmarked ones as `other`) and COUGHVID's and Coswara's coughs.
  - The **frame head** (`melati/nv_frames.py`, trained by `nonverbal/frames.py`) reads a whole segment and gives
    laughter, coughs, other vocal sounds and speech every 0.1 s; it tags **laughs** and **coughs**, including the ones
    over speech that no gap holds (most of AMI's). It trains on AMI's every frame and PodcastFillers' labelled regions.
  - Thresholds are chosen on PodcastFillers' validation episodes (`nonverbal/calib_pf.py`; coughs on AMI's dev
    meetings, as PodcastFillers labels none) and scored on its test episodes through the whole pipeline
    (`nonverbal/relabel.py`), on AMI's test meetings (`nonverbal/eval_ami.py`, `frames.py eval`) and, out of domain,
    on NonverbalTTS (`nonverbal/gap_rule.py`, `scripts/gap_curve.py`, `scripts/frames_diag.py`).
  - Released for non-commercial research as
    [Scicom-intl/Melati-NonVerbal-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-NonVerbal-Whisper-Large-V3):
    one checkpoint, the encoder and both heads, that `AutoModel` loads (`trust_remote_code`; `melati_hf/`), plus
    `melati_nonverbal.py` to tag a transcribed segment the way the pipeline does. Built and checked against the
    pipeline by `nonverbal/release.py`. NonverbalTTS's labels are CC BY-NC-SA 4.0, which cannot share one set of
    weights with PodcastFillers' licence, so it only scores the model; the first release trained on it and is replaced.
  - Rebuild: `deploy/box_nv_rel4.sh` (gap head), `box_nv_frames3.sh` (frame head), `box_nv_ship_rel4.sh` (into the
    pipeline, with the PodcastFillers and pilot re-runs).
- **Fillers**: written by the ASR model, or found when a voiced gap transcribed alone reads as one (Stage `fill`).

## What is wrong with Emilia-Pipe

`docs/emilia_bugs.md`: the defects of the upstream code, which ones the
[mesolitica fork](https://github.com/mesolitica/Emilia) fixed, and what each one does to the data, measured.

## Layout

| path | what |
|---|---|
| `melati/` | the pipeline stages and their shared code (`audio.py`, `tags.py`, `emotion.py`, `ser.py`, `nv_gaps.py`, `nonverbal.py`, `gaps.py`) |
| `asr/` | the Whisper fine-tune: corpus builders, mixes, training, merging, the release, the hallucination benchmark |
| `emotion/` | the emotion heads: corpus fetchers, manifest, features, training, the seven-model average, the labelling page |
| `nonverbal/` | the non-verbal heads: training clips (PodcastFillers, AMI, COUGHVID, Coswara), gap candidates, the gap and frame heads, their evaluations on NonverbalTTS, AMI and PodcastFillers |
| `serve/` | the model servers, the dynamic batcher, `launch.py` and the cluster config |
| `baselines/emilia_upstream/` | upstream Emilia-Pipe, unchanged, for the comparison |
| `scripts/` | experiments and audits that produce `results/` |
| `deploy/` | box scripts: environments, data fetches, runs; their settings in `deploy/env.sh` |
| `results/` | aggregate JSON the paper reads |
| `preprint/` | the paper |
| `tests/` | `pytest tests/` |

## Licences

The code is Apache-2.0 (`LICENSE`). Code from elsewhere keeps its own licence, named in each file's header:
`baselines/emilia_upstream/` is Amphion's Emilia-Pipe (MIT, its `LICENSE` beside it), and `melati/third_party/dnsmos.py`
is Microsoft's DNSMOS (CC BY 4.0). The dataset is CC BY-NC 4.0, as its source, and each released model's card states
its own licence.

## Citation

```bibtex
@misc{zolkepli2026melati,
  title  = {Melati: Multilingual Expressive speech, Labelled for Affect, Tags and Interjections},
  author = {Husein Zolkepli},
  year   = {2026},
  note   = {Scicom (MSC) Berhad}
}
```
