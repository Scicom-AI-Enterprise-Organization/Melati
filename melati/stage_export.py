"""Stage I: one row and one FLAC per segment. CPU.

Joins every stage's output for a segment, decides the tags under thresholds kept in ONE config
(export_config.json, written next to the manifest), and writes:
  <out>/audio/<rec>/<seg>.flac          24 kHz, 16-bit; the mix or the separated speech (below)
  <out>/manifest/part-<shard>.parquet   one row per segment, every measurement as a column
The rows are the dataset's (melati.pipeline.export_rows, which scripts/build_hf_dataset.py packs): the same tags, and
the same rows left out of the release (release_drop).

Nothing is filtered here. Quality scores, alignment confidence, the forced-cut and edge flags are
columns, and a subset is a query over them; Emilia applies its DNSMOS threshold before export, so
its choice cannot be revisited without re-running the corpus.

Audio: Emilia always exports the separated vocals. Here a segment keeps its original mix when
the separator found little to remove there (music at least BG_DB below the voice in the segment)
and SOTA-MOS does not prefer the separated version; otherwise the separated VOCAL version: speech
plus effects, music removed. Not the speech stem alone: DBS-TIGER files a laugh, a cough or a
breath under effects (scripts/nv_injection.py, results/separation/nv_injection.json: 98-100% of
the vocalisation's energy), so the speech stem would delete the sounds the tags describe.

Edges: the segment is trimmed to EDGE_PAD before its first sound (word or non-verbal event) and
after its last, never past the pauses Stage C cut in. A word that starts or ends within 30 ms of
the segment's own edge is flagged: that is what a clipped word looks like.

    python -m melati.stage_export --sources sources.jsonl --work $MELATI_DATA/podcast/work \\
        --out $MELATI_DATA/podcast/export --shard 0/8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from melati import audio as A
from melati import emotion as EMO
from melati.tags import Annotation, Event, Span, Word, is_filler, overlap_fraction, render
from melati import nv_frames as NF
from melati import nv_gaps as NG
from melati.nonverbal import TO_DELIVERY, TO_TAG

SR_OUT = 24000
# the AudioSet sounds Melati writes as point tags, and each one's threshold (audioset/sounds_vocab.py)
SOUNDS_INLINE = json.loads(Path(__file__).with_name("sounds_inline.json").read_text())
DEFAULTS = {
    "edge_pad_s": 0.15,
    "bg_db": -25.0,                # keep the mix when music sits this far below the voice
    "mix_mos_margin": 0.05,        # ...and SOTA-MOS does not prefer the separated speech by more
    # non-verbal tags come from Melati's own model on Whisper-large-v3, released as
    # Scicom-intl/Melati-NonVerbal-Whisper-Large-V3 (nonverbal/release.py), trained on PodcastFillers' CC BY episodes,
    # AMI (CC BY 4.0) and COUGHVID's and Coswara's coughs; not on NonverbalTTS (its labels are CC BY-NC-SA 4.0, which
    # cannot share weights with PodcastFillers' licence), VocalSound, or PodcastFillers' BY-SA / BY-ND episodes.
    # BREATHS from its gap head, in the word gaps (melati.nv_gaps), at the threshold chosen on PodcastFillers'
    # validation episodes for 0.8 precision where labelled (nonverbal/calib_pf.py, results/nonverbal/gap_eval_rel4pf.json;
    # its test episodes through the pipeline: 70% of the breaths found, 83% precise where labelled). LAUGHS and COUGHS
    # from its frame head (melati.nv_frames), over speech too, both at the thresholds chosen on AMI's dev meetings
    # (results/nonverbal/frames3_eval_pf.json; PodcastFillers labels no coughs). Laughs were at 0.9, chosen on
    # PodcastFillers' validation episodes for 0.8 precision; at 0.97 its test episodes through the pipeline read 60%
    # found and 87% precise where labelled, level with the head this one replaced (60%, 86%), where 0.9 read 71% and
    # 77% (results/nonverbal/frames_sweep_pf.json), and AMI's test meetings 29% found at 78% precision against that
    # head's 4%. A wrong [laughs] teaches a voice to laugh at the wrong moment; a missing one only teaches less. The
    # gap head's own laugh tags are not used: half of its extra ones in podcasts sound to AudioSet like gasps. A cluster without the model falls back to Qwen3-Omni (`omni`) for the gaps, read at ITS
    # thresholds (scripts/eval_nonverbal_gaps.py): laughs 91% / 18%, coughs 67% / 14%, no breaths. The PANNs events
    # tagged at 0.55 precision on PodcastFillers (scripts/eval_fillers.py), so they are no tag source unless
    # `sed_tags` is turned back on; they still gate separation (stage A).
    "nv_thresholds": {"heads": {"breath": 0.7}, "omni": {"laugh": 0.05, "cough": 0.95}},
    "nv_frame_thresholds": {"laugh": 0.97, "cough": 0.999},
    "sed_tags": False,
    "sed_scale": 1.0,              # x the detector's sensitive-profile thresholds (stored at 0.5)
    "sed_min_p": {},               # per-class probability floors, from scripts/eval_nonverbal.py
    # emotion: the average of seven models, Melati's five heads (melati/ser.py, Scicom-intl/Melati-Emotion-Heads:
    # trained on 21 corpora, retrained on 2026-11-04 after the box lost the first ones) and the two released
    # classifiers, and only the labels whose precision was measured on held-out speech (nEMO and NonverbalTTS's VoxCeleb
    # half, 7,791 clips no model trained on): each threshold is the lowest giving 0.75 precision at an assumed podcast
    # rate (results/emotion/retrain/ensemble_all.json; chosen on one half and measured on the other it holds 0.72-0.84,
    # recall happy 56%, sad 51%, angry 40%, surprised 13%, fearful 30%). Disgusted has no held-out positives and is
    # never tagged; every label's probability is still in the manifest. The published dataset was tagged by the first
    # heads at happy 0.51, angry 0.48 and fearful 0.44 (results/dataset/export_config.json).
    "emotion_models": ["h_whisper", "h_whisper_ft", "h_w2vbert", "h_wavlm", "h_e2v", "meralion", "e2v"],
    "emotion_thresholds": {"happy": 0.52, "sad": 0.64, "angry": 0.50, "surprised": 0.30, "fearful": 0.43},
    # where only some of the models tagged a segment, the thresholds measured the same way for that set: the two
    # released classifiers, alone and together (results/emotion/retrain/current.json), are what a cluster without the
    # five heads serves, and together they tag only happy, angry and surprised. Any other set is not tagged.
    "emotion_set_thresholds": {"meralion+e2v": {"happy": 0.88, "angry": 0.51, "surprised": 0.44},
                               "meralion": {"angry": 0.88, "surprised": 0.90}, "e2v": {"surprised": 0.49}},
    "emotion_default_threshold": None,  # a label without a measured threshold is not tagged
    "rate_spans": False,                # until fast/slow spans are checked by ear (melati.gaps.rate_spans)
    # rows left out of the release (release_drop). Tamil and Mandarin only where the ASR model heard them itself, in a
    # recording at least this share of whose transcribed segments it heard in that language: when the model detects
    # a language outside the allowed set, melati.pipeline.transcribe keeps the allowed language whose transcript scores
    # the best average log-probability, and on podcasts that lands on Tamil for Arabic chant, songs and stray Malay
    # (the release audit, scripts/release_language_fallback.py, found every Tamil row of the first publish wrong, and
    # most Mandarin ones). Malay and English chosen that way are kept: Whisper often hears Malay as Indonesian, and
    # their labels held up (about 99% right). Any row whose letters are mostly in a script its language is not
    # written in (Quranic Arabic labelled Malay or English), or with no text at all, is left out too.
    "minority_languages": {"ta": 0.5, "zh": 0.5},
    # AudioSet's frame-level taggers (nonverbal/audioset_pass.py: the average of PretrainedSED's BEATs, ATST-F and M2D,
    # fine-tuned on AudioSet's strong labels, MIT), for sounds where one of their thresholds holds on labelled speech
    # they never trained on (nonverbal/audioset_eval.py; results/nonverbal/audioset_BEATs+ATST-F+M2D.json). Each was
    # chosen on one half of NonverbalTTS's clips for 0.8 precision and measured on the other: sniffs at 0.15 (85 to 88%
    # precise, 68 to 71% found), throat clearing at 0.2 (81 to 83%, 34 to 47%), coughs at 0.5 (97%, 28%; beside the
    # frame head's, which finds none of NonverbalTTS's). Laughs at 0.3 beside the frame head's (0.97): on PodcastFillers'
    # test episodes through the pipeline the two together find 71% at 84% precise where labelled, the 0.9 frame head's
    # recall at 6 points more precision (results/nonverbal/frames_sweep_pf_audioset.json; on NonverbalTTS its laughs
    # alone hold 83 to 84% from 0.2). Whispering is a delivery over the words it covers, at 0.3:
    # on Expresso it covers 99% of the whispered utterances and none of the 10,000 in its other styles. Sighs, gasps,
    # sneezes, groans and the rest are not tagged: imprecise where labelled, never heard in podcasts, or (sighs) with
    # no labelled conversational speech to measure them on.
    "audioset_thresholds": {"laugh": 0.3, "sniff": 0.15, "throat_clear": 0.2, "cough": 0.5},
    "audioset_delivery": {"whisper": 0.3},
    # Melati's own AudioSet tagger (the sounds stage, Scicom-intl/Melati-AudioSet-EAT-Large): a row's `sounds` keeps
    # every label at this probability or more, highest first, for a reader to threshold label by label (the model
    # card lists each label's average precision on AudioSet's evaluation set). None where the stage has not run, which
    # is not the same as a segment where nothing was heard.
    "sounds_min_p": 0.1,
    # ...and as point tags in tagged_text ([music], [applause]): a class of the vocabulary v4 hears in the segment at
    # its gate or more, wherever the mean of the two frame models passes its threshold, the lowest at its floor or
    # more for 0.9 frame precision on AudioSet's strongly labelled test clips (audioset/sounds_vocab.py). Floor and
    # gate are 0.5, and 0.8 for Music: in the 5-hour pilot's listening check most [music] tags at 0.5 were faint
    # background music under the voice. Their precision on podcasts is not measured.
    "sounds_inline_gates": {k: v["gate"] for k, v in SOUNDS_INLINE["classes"].items()},
    "sounds_inline_low": 0.5,      # an event reaches as far as its frames stay over this share of the threshold
    "sounds_inline": {k: v["threshold"] for k, v in SOUNDS_INLINE["classes"].items()},
}

SCRIPTS = ("LATIN", "CJK", "TAMIL", "ARABIC", "THAI", "DEVANAGARI", "CYRILLIC", "HANGUL", "HIRAGANA", "KATAKANA")
WRITTEN_IN = {"ms": ("LATIN",), "en": ("LATIN",), "ta": ("TAMIL",), "zh": ("CJK",)}
SQUEEZED_S = 0.025      # a word the aligner gave less than this (one frame) has no audio of its own: text past the audio


def script_of(ch: str) -> str:
    import unicodedata
    name = unicodedata.name(ch, "")
    return next((s for s in SCRIPTS if name.startswith(s)), "OTHER")


def release_drop(row: dict, asr: dict | None, heard: dict, cfg: dict) -> str | None:
    """Why a row stays out of the release, or None to keep it. `heard`: the share of the recording's transcribed
    segments in which the ASR model detected each language itself."""
    lang = row["language"]
    share = cfg["minority_languages"].get(lang)
    if share is not None:
        if (asr or {}).get("detected_language") != lang:
            return "language_fallback"
        if heard.get(lang, 0.0) < share:
            return "language_stray"
    if not any(ch.isalnum() for ch in row["text"]):
        return "no_text"                 # the aligner placed none of the words; numbers read out are kept
    letters = [script_of(ch) for ch in row["text"] if ch.isalpha()]
    want = WRITTEN_IN.get(lang)
    if letters and want and sum(s in want for s in letters) * 2 < len(letters):
        return "wrong_script"
    return None


def load_jsonl(p: Path):
    return {r["seg"]: r for r in map(json.loads, p.open())} if p.exists() else {}


def stem_db(stems, key, a, b):
    """Mean power (dB) of one stem, or of several summed ("speech+sfx"), over [a, b) seconds."""
    fs = float(stems["frame_s"])
    i, j = int(a / fs), max(int(a / fs) + 1, int(b / fs))
    p = sum(10 ** (stems[k][i:j].astype(np.float32) / 10) for k in key.split("+"))
    if np.size(p) == 0:
        return float("-inf")
    return float(10 * np.log10(np.mean(p) + 1e-12))


def sound_events(snd, a, b, cfg):
    """The sounds stage's frames -> point-tag events in the cut's time ([a, b] is the cut in the segment's time). A
    class whose descendant in the vocabulary sounds over the same time gives way to it ([car alarm], not [car])."""
    fr = (snd or {}).get("frames") or {}
    th, step = cfg.get("sounds_inline") or {}, fr.get("step_s", 0.16)
    out = []
    for cls, series in fr.items():
        if cls not in th or th[cls] is None or snd["p"].get(cls, 0) < cfg["sounds_inline_gates"].get(cls, 1.0):
            continue
        for a_, b_, peak in NF.events_hysteresis(np.asarray(series, np.float32), th[cls],
                                                 th[cls] * cfg["sounds_inline_low"], min_frames=2, join=6):
            s_, e_ = a_ * step - a, b_ * step - a
            if e_ > 0 and s_ < b - a:
                out.append({"kind": SOUNDS_INLINE["classes"][cls]["tag"], "cls": cls, "s": round(max(s_, 0.0), 3),
                            "e": round(min(e_, b - a), 3), "p": round(peak, 4), "src": "sounds"})
    anc = {e["cls"]: set(SOUNDS_INLINE["classes"][e["cls"]]["ancestors"]) for e in out}
    return [e for e in out if not any(e["cls"] in anc[o["cls"]] and o["s"] < e["e"] and e["s"] < o["e"] for o in out)]


def build(seg, asr, al, mos, tag, fill, stems, cfg):
    """-> (row dict, (cut start, cut end) in recording time, audio version)."""
    words = [dict(w, src="align") for w in (al or {}).get("words") or []]
    s0 = seg["start"]
    # fillers recovered from voiced gaps become words with their own times
    for g in (fill or {}).get("gaps") or []:
        if g.get("kind") == "filler":
            t = g["text"].strip(" .,!?").lower() or "erm"
            words.append({"w": t, "s": g["s"], "e": g["e"], "c": 1.0, "filler": True, "src": "gap"})
    words.sort(key=lambda w: w["s"])
    # emotion phrases and rate spans index the ALIGNER's words; fillers inserted since shift them
    idx_map = [i for i, w in enumerate(words) if w["src"] == "align"]
    for w in words:
        w["filler"] = bool(w.get("filler")) or is_filler(w["w"])

    # non-verbal events: word-gap candidates labelled by the non-verbal heads (or Qwen3-Omni), each class at the
    # threshold chosen for that labeller
    events, delivery = [], []
    for c in (tag or {}).get("nv_gaps") or []:
        cls = NG.decide(c.get("dist") or {}, NG.thresholds_for(c, cfg["nv_thresholds"]))
        if cls in TO_TAG:
            p = max(v for k, v in c["dist"].items() if NG.TO_CLASS.get(k) == cls)
            events.append({"kind": TO_TAG[cls], "s": c["s"], "e": c["e"], "p": round(p, 4),
                           "src": f"{c.get('by', 'omni')}-gap"})
    # the frame-level detector's laughs and coughs (melati.nv_frames), where the gap labeller has none of the kind
    fr = (tag or {}).get("nv_frames") or {}
    for cls, t in (cfg.get("nv_frame_thresholds") or {}).items():
        if t is None or not fr.get(cls) or cls not in TO_TAG:
            continue
        fps, kind = fr.get("fps", 10), TO_TAG[cls]
        for a_, b_, peak in NF.events(np.asarray(fr[cls], np.float32), t):
            s_, e_ = a_ / fps, b_ / fps
            if any(e["kind"] == kind and e["s"] <= e_ + 0.3 and e["e"] >= s_ - 0.3 for e in events):
                continue
            events.append({"kind": kind, "s": round(s_, 3), "e": round(e_, 3), "p": round(peak, 4),
                           "src": "heads-frames"})
    # AudioSet's taggers (nonverbal/audioset_pass.py): the sounds the heads do not tag, and more of those they do, where
    # no event of the kind lies within 0.3 s; whispering as a delivery over the words it covers
    au = (tag or {}).get("audioset") or {}
    for cls, t in (cfg.get("audioset_thresholds") or {}).items() if au else ():
        if t is None or not au.get(cls) or cls not in TO_TAG:
            continue
        fps, kind = au.get("fps", 10), TO_TAG[cls]
        for a_, b_, peak in NF.events(np.asarray(au[cls], np.float32), t):
            s_, e_ = a_ / fps, b_ / fps
            if any(e["kind"] == kind and e["s"] <= e_ + 0.3 and e["e"] >= s_ - 0.3 for e in events):
                continue
            events.append({"kind": kind, "s": round(s_, 3), "e": round(e_, 3), "p": round(peak, 4),
                           "src": "audioset"})
    for cls, t in (cfg.get("audioset_delivery") or {}).items() if au and words else ():
        if t is None or not au.get(cls) or cls not in TO_DELIVERY:
            continue
        fps = au.get("fps", 10)
        for a_, b_, peak in NF.events(np.asarray(au[cls], np.float32), t):
            s_, e_ = a_ / fps, b_ / fps
            cover = [i for i, w in enumerate(words) if min(w["e"], e_) - max(w["s"], s_) > 0.5 * (w["e"] - w["s"])]
            if len(cover) >= 2:
                delivery.append(Span(TO_DELIVERY[cls], cover[0], cover[-1], round(peak, 4), "audioset"))
    # the detector's events and delivery spans, only if asked for (their precision was not good enough)
    floors = cfg["sed_min_p"]
    for e in ((tag or {}).get("sed") or []) if cfg["sed_tags"] else []:
        if e["p"] < floors.get(e["label"], 0.0):
            continue
        if e["label"] in TO_TAG:
            events.append({"kind": TO_TAG[e["label"]], "s": e["s"], "e": e["e"], "p": e["p"], "src": "panns"})
        elif e["label"] in TO_DELIVERY and words:
            cover = [i for i, w in enumerate(words) if min(w["e"], e["e"]) - max(w["s"], e["s"]) > 0.5 * (w["e"] - w["s"])]
            if len(cover) >= 2:
                delivery.append(Span(TO_DELIVERY[e["label"]], cover[0], cover[-1], e["p"], "sed"))

    # the cut: from the first sound to the last, never past Stage C's pauses
    sounds = [(w["s"], w["e"]) for w in words] + [(e["s"], e["e"]) for e in events]
    dur = seg["end"] - seg["start"]
    if sounds:
        a = max(0.0, min(s for s, _ in sounds) - cfg["edge_pad_s"])
        b = min(dur, max(e for _, e in sounds) + cfg["edge_pad_s"])
    else:
        a, b = 0.0, dur
    for w in words:
        w["s"], w["e"] = round(w["s"] - a, 3), round(w["e"] - a, 3)
    for e in events:
        e["s"], e["e"] = round(e["s"] - a, 3), round(e["e"] - a, 3)
    # AudioSet sounds as point tags, added after the cut so a sound never widens it
    events += sound_events((tag or {}).get("sounds"), a, b, cfg)

    ann = Annotation(words=[Word(w["w"], w["s"], w["e"], float(w.get("c", 1.0)), w["filler"]) for w in words])
    for e in events:
        ev = Event(e["kind"], e["s"], e["e"], e["p"], e.get("src", ""))
        ev.overlap = round(overlap_fraction(ev, [x for x in ann.words if not x.filler]), 3)
        ann.events.append(ev)
    ann.delivery = delivery

    # emotion: average the chosen models' phrase probabilities, then threshold them at what was measured for that set
    emo = (tag or {}).get("emotion") or {}
    models = [m for m in cfg["emotion_models"] if m in emo]
    thr = (cfg["emotion_thresholds"] if models == list(cfg["emotion_models"])
           else (cfg.get("emotion_set_thresholds") or {}).get("+".join(models), {}))
    seg_probs = None
    if models and words:
        base = {tuple(p[:2]): (p[2], p[3], np.array(p[4])) for p in emo[models[0]]["phrases"]}
        for m in models[1:]:
            for p in emo[m]["phrases"]:
                if tuple(p[:2]) in base:
                    s_, e_, v = base[tuple(p[:2])]
                    base[tuple(p[:2])] = (s_, e_, v + np.array(p[4]))
        phr = sorted(base.items())
        spans = EMO.spans_from_phrases([(a_, b_, s_, e_) for (a_, b_), (s_, e_, _) in phr],
                                       [v / len(models) for _, (_, _, v) in phr], thr,
                                       cfg["emotion_default_threshold"])
        for lab, f, l, conf in spans:
            if f < len(idx_map) and l < len(idx_map):
                ann.emotion.append(Span(lab, idx_map[f], idx_map[l], conf, "+".join(models)))
        segp = [np.array(emo[m]["segment"]) for m in models if emo[m].get("segment")]
        if segp:
            seg_probs = np.mean(segp, 0)

    if cfg["rate_spans"]:
        for lab, f, l in (tag or {}).get("rate") or []:
            if f < len(idx_map) and l < len(idx_map):
                ann.delivery.append(Span(lab, idx_map[f], idx_map[l], 1.0, "rate"))

    # the mix, or the separated vocal version (speech + effects: only the music removed)
    t_a, t_b = s0 + a, s0 + b
    bg = stem_db(stems, "music", t_a, t_b) - stem_db(stems, "speech+sfx", t_a, t_b)
    mos = dict(mos or {})
    ms = mos.get("mos_vocal", mos.get("mos_speech"))          # mos.jsonl written before the vocal version
    mm = mos.get("mos_mix")
    use_mix = bg <= cfg["bg_db"] and (ms is None or mm is None or mm >= ms - cfg["mix_mos_margin"])
    version = "mix" if use_mix else "vocal"
    dn_v = mos.get("dnsmos_vocal", mos.get("dnsmos_speech")) or {}

    snd = (tag or {}).get("sounds")
    plain = render(ann, emotions=False, delivery=False, events=False)
    row = {
        "id": seg["seg"], "recording": seg["rec"], "speaker": f"{seg['rec']}_{seg['speaker']}",
        "start": round(t_a, 3), "end": round(t_b, 3), "duration": round(b - a, 3),
        "language": (asr or {}).get("language"), "text": plain, "asr_text": (asr or {}).get("text"),
        "tagged_text": render(ann), "tagged_text_hesitates": render(ann, fillers_as="tag"),
        "annotation": json.dumps(ann.to_json(), ensure_ascii=False),
        "emotion_segment": (EMO.LABELS[int(np.argmax(seg_probs))] if seg_probs is not None else None),
        "emotion_segment_p": (round(float(np.max(seg_probs)), 4) if seg_probs is not None else None),
        "sounds": ([{"label": k, "p": v} for k, v in snd["p"].items() if v >= cfg["sounds_min_p"]] if snd else None),
        "n_events": sum(e.source != "sounds" for e in ann.events),
        "n_sound_tags": sum(e.source == "sounds" for e in ann.events), "n_fillers": sum(w.filler for w in ann.words),
        "n_dropped_words": sum(g.get("kind") == "word" for g in (fill or {}).get("gaps") or []),
        "squeezed_words": int(sum(w.end - w.start < SQUEEZED_S for w in ann.words)),    # numpy times give int64
        "audio_version": version, "background_db": round(bg, 1) if np.isfinite(bg) else None,
        "mos": (mm if use_mix else ms), "mos_vocal": ms, "mos_mix": mm,
        "dnsmos": ((mos.get("dnsmos_mix") or {}) if use_mix else dn_v).get("OVRL"),
        "align_score": (al or {}).get("score"),
        "edge_clip_start": bool(al and al.get("words") and al["words"][0]["s"] < 0.03),
        "edge_clip_end": bool(al and al.get("tail_s") is not None and al["tail_s"] < 0.03),
        "forced_cut": seg.get("forced_cut", False), "overlap_s_removed": seg.get("overlap_s_removed", 0.0),
        "asr_avg_logprob": (asr or {}).get("avg_logprob"), "asr_no_speech_prob": (asr or {}).get("no_speech_prob"),
        "asr_fallback_t": (asr or {}).get("fallback_t"), "asr_looped": (asr or {}).get("looped"),
        "prev": seg.get("prev"), "next": seg.get("next"),
    }
    return row, (t_a, t_b), version


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--config", type=Path, default=None, help="overrides for DEFAULTS")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from melati.pipeline import export_rows

    cfg = dict(DEFAULTS, **(json.loads(args.config.read_text()) if args.config else {}))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "export_config.json").write_text(json.dumps(cfg, indent=1))
    i, n = map(int, args.shard.split("/"))
    srcs = [json.loads(l) for l in args.sources.open()]
    todo = [r for k, r in enumerate(srcs) if k % n == i and (args.work / r["id"] / "align.jsonl").exists()]
    if args.limit:
        todo = todo[: args.limit]
    rows, dropped = [], {}
    for k, rec in enumerate(todo, 1):
        for row, x in export_rows(args.work, rec, cfg, dropped=dropped):
            A.write_flac(args.out / row["audio"], x, SR_OUT)
            rows.append(row)
        if k % 100 == 0:
            print(f"[export] {k}/{len(todo)} recordings, {len(rows)} segments", flush=True)
    (args.out / "manifest").mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out / "manifest" / f"part-{i:03d}.parquet", compression="zstd")
    left = {why: f"{r} rows, {s / 3600:.2f} h" for why, (r, s) in sorted(dropped.items())}
    print(f"[export] shard {i}/{n}: {len(rows)} segments; left out of the release: {left or 'none'}", flush=True)


if __name__ == "__main__":
    main()
