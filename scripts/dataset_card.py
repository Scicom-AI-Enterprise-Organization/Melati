#!/usr/bin/env python3
"""The dataset card and its figures, from what scripts/build_hf_dataset.py measured (<hf>/stats.json) and the
thresholds the export used (<hf>/export_config.json). Runs anywhere (matplotlib only).

    python scripts/dataset_card.py --hf $MELATI_DATA/podcast/hf --repo Scicom-intl/Melati
"""
import os
import argparse
import json
import sys
from pathlib import Path

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

CODE = "https://github.com/Scicom-AI-Enterprise-Organization/Melati"
ASR = "Scicom-intl/Melati-Whisper-Large-V3"
EMO = "Scicom-intl/Melati-Emotion-Whisper-Large-V3"
NV = "Scicom-intl/Melati-NonVerbal-Whisper-Large-V3"
AS = "Scicom-intl/Melati-AudioSet-EAT-Large"
TIMING = Path(f"{MELATI_DATA}/results/audioset/timing_eval.json")    # audioset/timing_eval.py
LANG = {"ms": "Malay", "en": "English", "ta": "Tamil", "zh": "Mandarin", "none": "none"}
EVENT_NAMES = (("breathes", "breaths"), ("laughs", "laughs"), ("coughs", "coughs"), ("sniffs", "sniffs"),
               ("clears throat", "throat\nclearing"))
AUDIOSET = "BEATs+ATST-F+M2D"           # nonverbal/audioset_pass.py: PretrainedSED's three, averaged
EMOTIONS = ("happy", "sad", "angry", "surprised", "fearful")


def pf_numbers(cfg):
    """The released non-verbal model on PodcastFillers' test episodes through the pipeline, from the results files its
    own card reads (nonverbal/release.py; the versions deploy/box_nv_ship_rel4.sh shipped), laughs at the threshold
    this export tags them at (nonverbal/sweep_frames.py). RUNS ON THE BOX."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nonverbal"))
    from release import Numbers
    n = Numbers("rel4", 3, "rel4f")
    out = {c: n.pf_class(n.pf, c) for c in ("laughter", "breath")}
    t, u = cfg["nv_frame_thresholds"]["laugh"], (cfg.get("audioset_thresholds") or {}).get("laugh")
    sweep = Path(f"{MELATI_DATA}/results/nonverbal/" + ("frames_sweep_pf_audioset.json" if u else "frames_sweep_pf.json"))
    at = json.loads(sweep.read_text())["laugh"].get(f"{t}+audioset_{u}" if u else str(t)) if sweep.exists() else None
    if at:
        out["laughter"] = (at["recall"], at["precision_labelled"])
    return out


def audioset_numbers(cfg):
    """AudioSet's taggers at the export's thresholds (nonverbal/audioset_eval.py): on NonverbalTTS, each half of its
    clips measured at the threshold the other half chose, and whispering on Expresso's styles. RUNS ON THE BOX."""
    f = Path(f"{MELATI_DATA}/results/nonverbal/audioset_{AUDIOSET}.json")
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    names = {"sniff": "sniffs", "throat_clear": "clears throat", "cough": "coughs", "laugh": "laughs"}
    out = {}
    for cls, t in (cfg.get("audioset_thresholds") or {}).items():
        halves = d["nvtts"].get(names.get(cls, cls) + "/halves")
        if halves:
            ms = [h[str(t)] for h in halves]
            out[cls] = {"p": [m["precision"] for m in ms], "r": [m["recall"] for m in ms], "ref": sum(m["ref"] for m in ms)}
    w = (cfg.get("audioset_delivery") or {}).get("whisper")
    cov = d.get("expresso", {}).get("covers_half", {}).get("whisper", {}).get(str(w)) if w else None
    if cov:
        out["whisper"] = {"found": cov["whisper"], "others": max(v for k, v in cov.items() if k != "whisper"),
                          "n_others": sum(n for k, n in d["expresso"]["utterances"].items() if k != "whisper")}
    return out


def figures(out: Path, st: dict, repo: str) -> dict:
    """The card's charts, one per file, in the preprint's style (melati/plotstyle.py); -> {name: the <img> that shows
    it on the card at the figures' common scale}."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from melati.plotstyle import CARD_DPI, MUTED, S1, S2, S4, S5, TEXT, better, card_img, frame, hf_url, legend_below
    import matplotlib.pyplot as plt
    thousands = plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:.0f}")
    img = {}

    def save(fig, name, alt):
        (out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "figures" / name, dpi=CARD_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        img[name.removesuffix(".png")] = card_img(out / "figures" / name, alt,
                                                  hf_url(repo, f"figures/{name}", "dataset"))
    # 1. hours per language
    langs = [k for k in st["hours_by_language"] if k != "none"]
    hs = [st["hours_by_language"][k] for k in langs]
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    frame(ax, None, "hours", axis="y")
    ax.bar(range(len(langs)), hs, 0.55, color=S1, zorder=3)
    for i, v in enumerate(hs):
        lab = f"{v:,.0f} h" if v >= 10 else f"{v:.1f} h" if v >= 0.1 else f"{60 * v:.0f} min"
        ax.text(i, v, lab, ha="center", va="bottom", fontsize=6.8, color=TEXT)
    ax.set_xticks(range(len(langs))); ax.set_xticklabels([LANG.get(k, k) for k in langs])
    save(fig, "languages.png", "Hours per language")
    # 2. segment lengths
    hist = st["duration_hist"]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    frame(ax, "segment length (s)", "segments", axis="y")
    ax.bar([i + 0.5 for i in range(len(hist))], hist, 0.9, color=S1, zorder=3)
    ax.set_xlim(0, 30)
    ax.yaxis.set_major_formatter(thousands)
    save(fig, "lengths.png", "Segment lengths")
    # 3. tags per hour of exported speech, coloured as the preprint's: non-verbal sounds, fillers, emotion spans
    names = [n for _, n in EVENT_NAMES] + ["whispered\nspans", "fillers"] + list(EMOTIONS)
    vals = [st["events_per_hour"].get(k, 0.0) for k, _ in EVENT_NAMES]
    vals += [(st.get("delivery_per_hour") or {}).get("whisper", 0.0), st["fillers_per_hour"]]
    vals += [st["emotion_spans"].get(e, 0) / st["hours"] for e in EMOTIONS]
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    frame(ax, None, "per hour of speech", axis="y")
    ax.bar(range(len(names)), vals, 0.62, color=[S1] * len(EVENT_NAMES) + [S5, S2] + [S4] * len(EMOTIONS), zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=6.8, color=TEXT)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    save(fig, "tags_per_hour.png", "Tags per hour of speech")
    # 4. quality scores
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    frame(ax, better("score of the segment as exported", lower=False), "segments")
    xs = [1.05 + 0.1 * i for i in range(40)]
    for key, c, lab in (("mos_hist", S1, "SOTA-MOS"), ("dnsmos_hist", S4, "DNSMOS (Emilia's filter)")):
        ax.plot(xs, st[key], color=c, linewidth=1.6, label=lab, zorder=3)
    ax.axvline(3.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(3.7, ax.get_ylim()[1] * 0.9, "Emilia-Pipe keeps\nDNSMOS \u2265 3 (dashed)", fontsize=6.6, color=MUTED, va="top")
    ax.set_xlim(1, 5)
    ax.yaxis.set_major_formatter(thousands)
    legend_below(ax, ncol=2, pad=0.3)
    save(fig, "quality.png", "SOTA-MOS and DNSMOS of every segment")
    return img


COLUMNS = [
    ("audio", "the segment, 24 kHz mono FLAC (16-bit), cut from its first sound to its last plus 0.15 s"),
    ("id, recording, speaker", "segment id; source recording; speaker within that recording (diarized per "
                               "recording, so one person in two recordings has two ids; a recording over 45 minutes "
                               "(80 for those finished before 2026-10-26), or whose single pass failed, is diarized in "
                               "ten-minute windows and its ids, `<recording>_w<k>S<j>`, are per window)"),
    ("start, end, duration", "where the segment sits in the source recording, in seconds"),
    ("language", "the language the transcript was decoded in (`ms`, `en`, `ta`, `zh`; Tamil and Mandarin only where "
                 "the ASR model heard them itself, in a recording mostly in that language)"),
    ("text", "the transcript as words: fillers kept as spoken (`um`, `erm`, `emm`), no tags"),
    ("tagged_text", "the transcript with tags in Fish Audio Drama 3's syntax: emotion scopes, non-verbal point tags "
                    "and AudioSet sound tags (`[music]`, `[applause]`)"),
    ("tagged_text_hesitates", "the same with each filler written as `[hesitates]`"),
    ("annotation", "JSON: every word with its time and confidence, every non-verbal event with its time, "
                   "probability and source, every emotion span with its words and probability. The tagged text is "
                   "a rendering of this"),
    ("asr_text", "the transcript as the ASR model wrote it, before fillers found in the gaps were added"),
    ("emotion_segment, emotion_segment_p", "the most probable emotion of the whole segment (the seven-model "
                                           "average) and its probability, whatever the threshold"),
    ("sounds", "every AudioSet label Melati's own tagger (Melati-AudioSet-EAT-Large) gives the segment at 0.1 or more, "
               "highest first, as `{label, p}`; each label's average precision on AudioSet is on the model's card. "
               "Speech is on nearly every row"),
    ("n_events, n_breaths, n_laughs, n_coughs, n_sniffs, n_throat_clears, n_fillers", "counts, for filtering"),
    ("n_sound_tags", "AudioSet sound tags in the tagged text"),
    ("whispered", "the segment holds a whispered span (`<whisper>...</whisper>` in the tagged text)"),
    ("n_dropped_words", "voiced gaps between aligned words that, transcribed alone, came back as a word the transcript "
                        "does not have: mostly backchannels (`ha`, `ya`, `okay`) and short function words written for "
                        "a breath or a click, so a count to filter on, not words missing from the text"),
    ("squeezed_words", "words the aligner fit into under 25 ms (one frame): text the audio does not hold, usually "
                       "words the ASR model wrote past the end of what was cut. `squeezed_words < 2` is the stricter set"),
    ("audio_version", "`mix` (the recording itself) or `vocal` (music removed by DBS-TIGER, speech and "
                      "vocal sounds kept), whichever SOTA-MOS prefers where there was music"),
    ("mos, mos_vocal, mos_mix", "SOTA-MOS of the exported audio, of the separated version and of the mix"),
    ("dnsmos", "DNSMOS P.835 overall score of the exported audio (Emilia's filter)"),
    ("lufs, peak_db, background_db", "loudness, peak, and music level against speech in the segment"),
    ("align_score, edge_clip_start, edge_clip_end", "forced-alignment confidence; a word starting or ending "
                                                    "within 30 ms of the cut"),
    ("forced_cut, overlap_s_removed", "a segment cut inside speech because it ran past 30 s; seconds of "
                                      "overlapping speech removed"),
    ("asr_avg_logprob, asr_fallback_t, asr_looped", "decoder confidence; the temperature a looping transcript "
                                                    "was re-decoded at (0 when greedy was kept); a loop survived"),
    ("prev, next", "the neighbouring segments of the same recording, for context"),
    ("source, source_title, youtube_id", "where the audio comes from"),
]


def card(st: dict, cfg: dict, repo: str, pf: dict, img: dict, asn: dict) -> str:
    langs = [k for k in st["hours_by_language"] if k != "none"]
    hours = st["hours"]
    lang_yaml = "\n".join(f"- {k}" for k in langs)
    n = st["segments"]
    size = "1M<n<10M" if n >= 1e6 else "100K<n<1M" if n >= 1e5 else "10K<n<100K"
    hm = lambda h: f"{h:,.1f}" if h >= 1 else f"{60 * h:.0f} min"  # noqa: E731
    lang_rows = "\n".join(f"| {LANG.get(k, k)} | {hm(st['hours_by_language'][k])} | "
                          f"{st['segments_by_language'][k]:,} |" for k in langs)
    ev = st["events_per_hour"]
    au_t, au_d = cfg.get("audioset_thresholds") or {}, cfg.get("audioset_delivery") or {}
    rng = lambda xs: (f"{100 * min(xs):.0f}%" if round(100 * min(xs)) == round(100 * max(xs))  # noqa: E731
                      else f"{100 * min(xs):.0f} to {100 * max(xs):.0f}%")
    au_line = "; ".join(f"{lab} {rng(asn[c]['p'])} precise, {rng(asn[c]['r'])} found" for c, lab in
                        (("sniff", "sniffs"), ("throat_clear", "throat clearing"), ("cough", "coughs"),
                         ("laugh", "laughs")) if c in asn)
    wh = asn.get("whisper")
    emo_t = cfg["emotion_thresholds"]
    nv_t, fr_t = cfg["nv_thresholds"]["heads"], cfg["nv_frame_thresholds"]
    cols = "\n".join(f"| `{a}` | {b} |" if "," not in a else
                     f"| {', '.join(f'`{x.strip()}`' for x in a.split(','))} | {b} |" for a, b in COLUMNS)
    ef = st["emilia_filter"]
    example = (st.get("examples") or [{"id": "none", "tagged_text": ""}])[0]
    partial = "" if st.get("complete") else (
        f"> **Work in progress.** A partial release: it holds {st['release_recordings']:,} of the source's "
        f"{st['source_recordings']:,} recordings, {st['blocks_packed']} of its {st['blocks']} shards, and the rest "
        "will be added in later releases. The columns, tags and numbers may change as it grows; every number on "
        "this card is over what is here.\n\n")
    pc = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    lo = st.get("left_out") or {}
    left_n = sum(v["segments"] for v in lo.values())
    left_h = (lambda h: f"{h:,.1f} h" if h >= 1 else f"{60 * h:.0f} min")(sum(v["hours"] for v in lo.values()))
    why = {"language_fallback": "Tamil or Mandarin the ASR model did not hear itself: when it detects a language "
                                "outside Malay, English, Mandarin and Tamil, the segment is decoded in each of those and "
                                "the best-scoring kept, and on podcasts that choice lands on Tamil for Arabic chant, "
                                "songs and stray Malay",
           "language_stray": "Tamil or Mandarin heard in a recording mostly in another language: songs, chants and "
                             "a stray phrase",
           "wrong_script": "text whose letters are mostly in a script its language is not written in",
           "no_text": "no text at all: the aligner placed none of the words"}
    left = ("**Left out**, as a transcript that cannot be what was said:\n\n| why | segments | hours |\n|---|---|---|\n"
            + "\n".join(f"| {why.get(k, k)} | {v['segments']:,} | {hm(v['hours'])} |" for k, v in lo.items())
            ) if lo else ""
    fm = json.loads(TIMING.read_text())["frame_map"] if TIMING.exists() else {}
    timing_line = (f" (frame mAP {fm['mean of the two frame models']:.3f} on AudioSet's time-stamped test clips, where "
                   f"the tagger alone, read off short windows, reaches {fm['v4, 2 s windows']:.3f})" if fm else "")
    sn = st.get("sounds") or {"music_rows": 0, "music_hours": 0.0, "music_by_audio_version": {}}
    sph = st.get("sound_tags_per_hour") or {}
    sound_line = ", ".join(f"`[{k}]` {v:.2f} an hour" if v >= 0.01 else f"`[{k}]` {st['sound_tags'][k]:,} in all"
                           for k, v in sorted(sph.items(), key=lambda kv: -kv[1])) or "none yet"
    w = st.get("windowed")
    windowed = (f", or per ten-minute window in the {w['recordings']:,} recordings ({w['hours']:,.0f} h) diarized "
                "in windows" if w and w["recordings"] else "")
    return f"""---
license: cc-by-nc-4.0
language:
{lang_yaml}
pretty_name: Melati
size_categories:
- {size}
task_categories:
- text-to-speech
- automatic-speech-recognition
- audio-classification
tags:
- speech
- emotion
- non-verbal
- laughter
- breath
- filler
- audioset
- podcast
- malaysian
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Melati

{partial}**M**ultilingual **E**xpressive speech, **L**abelled for **A**ffect, **T**ags and **I**nterjections: {hours:,.0f} hours
of Malaysian podcast speech in {n:,} segments, each with its transcript, word times, speaker, quality scores and
tags for **emotion**, **non-verbal sounds** (breaths, laughs, coughs, sniffs, throat clearing), **whispering**,
**fillers** and the **sounds** around the speech (music, clapping, applause), written in
[Fish Audio Drama 3](https://huggingface.co/spaces/fishaudio/Drama-3)'s tag syntax:

```
{example['tagged_text']}
```

(segment `{example['id']}`.) Built by [Melati]({CODE}), an in-the-wild speech-data pipeline in the line of
[Emilia-Pipe](https://arxiv.org/abs/2501.15907), from the recordings of
[{st['source']}](https://huggingface.co/datasets/{st['source']}), the podcasts behind
[Malaysian-Emilia](https://huggingface.co/datasets/mesolitica/Malaysian-Emilia). No score is filtered on:
every measurement is a column, and a subset is a query (below). The only rows left out are those whose transcript
cannot be what was said ({left_n:,} segments, {left_h}: below the languages).

| | |
|---|---|
| source recordings | {st['release_recordings']:,} of {st['source_recordings']:,} ({st['release_source_hours']:,.0f} h; {st['empty_source_files']} of the source's files are empty) |
| segments | {n:,} |
| hours | {hours:,.1f} |
| speaker ids | {st['speakers']:,}, per recording{windowed} |
| audio | 24 kHz mono FLAC, inline in the parquet shards |

| language | hours | segments |
|---|---|---|
{lang_rows}

{left}

{img['languages']}

{img['lengths']}

## Use

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train", streaming=True)
for row in ds:
    audio = row["audio"]["array"]           # 24 kHz
    print(row["tagged_text"], row["language"], row["mos"])
    break
```

Subsets are filters on the columns. The core of Emilia-Pipe's filter (3 to 30 s, DNSMOS at least 3) keeps
{ef['segments']:,} segments, {ef['hours']:,.1f} h:

```python
emilia_like = ds.filter(lambda r: 3 <= r["duration"] <= 30 and (r["dnsmos"] or 0) >= 3)
laughs = ds.filter(lambda r: r["n_laughs"] > 0)
no_music = ds.filter(lambda r: not any(s["label"] == "Music" and s["p"] >= 0.5 for s in r["sounds"] or []))
```

## Tags

{img['tags_per_hour']}

*Per hour of speech: non-verbal sounds (blue), fillers (teal), emotion spans (amber).*

Every tag is a rendering of time-stamped events kept in `annotation`, so a different target format needs no
re-annotation. A tag class appears only where its precision was measured on labelled speech no model trained on,
at a threshold chosen on other labelled speech; every threshold is in `export_config.json`.

- **Emotion** spans (`<happy>...</happy>`; neutral is untagged): the average of seven models, Melati's heads on five
  frozen encoders ([one released]({'https://huggingface.co/' + EMO})) and MERaLiON-SER and emotion2vec+, over
  phrases between pauses. Thresholds (happy {emo_t['happy']}, sad {emo_t['sad']}, angry {emo_t['angry']},
  surprised {emo_t['surprised']}, fearful {emo_t['fearful']}) give 0.75 precision at a podcast's assumed label rate
  on held-out English and Polish speech (nEMO, NonverbalTTS); chosen on one half and measured on the other they hold
  0.74 to 0.82. Disgust is never tagged.
- **Non-verbal** point tags. Breaths (`[breathes]`, {ev.get('breathes', 0):.1f} an hour) come from
  [{NV}](https://huggingface.co/{NV})'s gap head in the gaps between aligned words (threshold {nv_t['breath']});
  laughs (`[laughs]`, {ev.get('laughs', 0):.1f} an hour, {st['laughs_over_speech']:,} of them over speech) and coughs
  (`[coughs]`, {ev.get('coughs', 0):.2f} an hour) from its frame head, every 0.1 s, over speech too (laugh
  {fr_t['laugh']}, cough {fr_t['cough']}), and from AudioSet's taggers (laugh {au_t.get('laugh')}, cough
  {au_t.get('cough')}); sniffs (`[sniffs]`, {ev.get('sniffs', 0):.2f} an hour, threshold {au_t.get('sniff')}) and
  throat clearing (`[clears throat]`, {ev.get('clears throat', 0):.2f} an hour, {au_t.get('throat_clear')}) from
  AudioSet's taggers alone: the average of [PretrainedSED](https://github.com/fschmid56/PretrainedSED)'s BEATs, ATST-F
  and M2D, fine-tuned on AudioSet's time-stamped labels, every 0.1 s. On PodcastFillers' test episodes through this
  pipeline the tags find {pc(pf['breath'][0])} of breaths and {pc(pf['laughter'][0])} of laughs,
  {pc(pf['breath'][1])} and {pc(pf['laughter'][1])} precise where labelled. AudioSet's thresholds were each chosen on
  one half of NonverbalTTS's clips (interviews and expressive speech no model here trained on) and measured on the
  other: {au_line}.
- **Delivery**: whispering (`<whisper>...</whisper>`, {(st.get('delivery_per_hour') or {}).get('whisper', 0):.2f}
  spans an hour) over the words AudioSet's taggers hear whispered (threshold {au_d.get('whisper')}){f": on Expresso they cover {pc(wh['found'])} of the whispered utterances and {pc(wh['others'])} of the {wh['n_others']:,} in its other styles" if wh else ""}.
- **Fillers** are words (`um`, `erm`, `emm`, {st['fillers_per_hour']:.0f} an hour), written by the ASR model or
  recovered from a voiced gap transcribed alone; `tagged_text_hesitates` writes them as `[hesitates]`.
- **Sounds** around the speech ({sound_line}): [{AS}](https://huggingface.co/{AS}), Melati's own AudioSet tagger,
  says which sounds a segment holds, and the mean of PretrainedSED's taggers and the frame head trained beside it
  says where{timing_line}. {len(cfg['sounds_inline'])} sounds can be written, each at the lowest threshold of 0.5 or more for
  0.9 precision on those clips and only where the tagger hears it in the segment at 0.5 or more. Music needs 0.8 on
  both: below that it was mostly faint music under the voice. The voice itself (speech, laughs, breaths, coughs) is
  left to the tags above, and so are the sounds of a room or a channel. Their precision on podcasts is not measured.

## Quality

{img['quality']}

`sounds` is the place to filter on what else is in a segment. It hears music at 0.5 or more in
{sn['music_rows']:,} segments ({sn['music_hours']:,.1f} h), {sn['music_by_audio_version'].get('mix', 0):,} of them
exported as the recording itself: the separator runs only where PANNs hears background, and in these it heard none,
so they keep their music, mostly faint under the voice.

Each segment is exported from the recording itself, or, where the separator heard music, from the separated version
(speech and vocal sounds kept, music removed: a speech-only stem would delete the laughs and breaths the tags
describe), whichever SOTA-MOS scores higher. {st['audio_version'].get('vocal', 0):,} segments are the separated
version. SOTA-MOS median {st['mos'].get('50')}, DNSMOS median {st['dnsmos'].get('50')}.

## Columns

| column | what |
|---|---|
{cols}

## How it was built

| stage | model |
|---|---|
| decode, one gain per recording; music separation only where PANNs hears background | PANNs Cnn14, [DBS-TIGER](https://huggingface.co/JusperLee/DBS-TIGER-DnR) |
| who spoke when | [Who-Spoke-When-Over-the-Phone](https://huggingface.co/Scicom-intl/Who-Spoke-When-Over-the-Phone-Whisper-Large-V3-Merged) |
| single-speaker segments, cut only in pauses | Silero VAD |
| transcription | [{ASR}](https://huggingface.co/{ASR}) |
| word times | [Where-Words-Begin-and-End](https://huggingface.co/Scicom-intl/Where-Words-Begin-and-End-Whisper-Large-V3) |
| quality | SOTA-MOS (24 kHz), DNSMOS P.835 |
| sniffs, throat clearing, whispering, more laughs and coughs | [PretrainedSED](https://github.com/fschmid56/PretrainedSED)'s BEATs, ATST-F and M2D on AudioSet's strong labels, averaged |
| emotion, non-verbal, fillers | above |

The code, the evaluation of every model and the comparison with Emilia-Pipe on the same recordings are in
[Melati on GitHub]({CODE}).

## Limitations

- The transcripts, speakers, tags and scores are model output. Every tag's precision was measured on English (and
  Polish, for emotion) speech; on Malay speech it is not yet measured against human labels.
- Coughs, sniffs, throat clearing and whispering are rare in podcasts and rarer as tags: each is tagged only where
  its precision held, so most are missed. Sighs, gasps, crying, humming, singing and shouting are not tagged yet:
  AudioSet's taggers were imprecise on them, never heard them in podcasts, or (sighs) there is no labelled
  conversational speech to measure them on.
- Speaker ids hold within one recording only, and within one ten-minute window in a recording diarized in windows.

## Licence

CC BY-NC 4.0, as the source. The audio belongs to its owners; it is shared for non-commercial research, under the
source's own notice:

> All the videos, songs, images, and graphics used in the video belong to their respective owners and I does not
> claim any right over them. Copyright Disclaimer under section 107 of the Copyright Act of 1976, allowance is made
> for "fair use" for purposes such as criticism, comment, news reporting, teaching, scholarship, education and
> research.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--repo", default="Scicom-intl/Melati")
    args = ap.parse_args()
    st = json.loads((args.hf / "stats.json").read_text())
    cfg = json.loads((args.hf / "export_config.json").read_text())
    img = figures(args.hf, st, args.repo)
    (args.hf / "README.md").write_text(card(st, cfg, args.repo, pf_numbers(cfg), img, audioset_numbers(cfg)))
    print(f"card -> {args.hf / 'README.md'}")


if __name__ == "__main__":
    main()
