#!/usr/bin/env python3
"""The \\val{} numbers (numbers.tex) and abstract.txt for the preprint, from result files only.

    ../.venv/bin/python make_tables.py

Reads the aggregate JSON under ../results/ (pulled from the box); nothing here reads audio,
transcripts of real recordings or per-file predictions. A number that is not in a result file yet
is simply absent, and LaTeX prints ?? for it, so a pending result cannot pass for a measured one.
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
NUM = {}


def J(rel):
    p = RES / rel
    return json.loads(p.read_text()) if p.exists() else None


def put(k, v):
    if v is not None:
        NUM[k] = v


def pct(v, nd=0):
    return None if v is None else f"{100 * v:.{nd}f}\\%"


def fx(v, nd=3):
    return None if v is None else f"{v:.{nd}f}"


def audit():
    for k in ("v1", "v2"):
        d = J(f"emilia/audit_emilia_{k}.json")
        if not d:
            continue
        put(f"audit/{k}/files", f"{d['files']:,}")
        put(f"audit/{k}/kbps", f"{d['bitrate_kbps']['p50']:.0f}")
        put(f"audit/{k}/cutoff", f"{d['cutoff_khz']['50']:.1f}")
        put(f"audit/{k}/fullscale", pct(d["share_peak_at_full_scale"]))
        put(f"audit/{k}/peak95", f"{d['peak_db']['95']:+.2f}")


def separation():
    d = J("separation/nv_injection.json")
    if not d:
        return
    none = {k.split("/")[0]: v for k, v in d.items() if k.endswith("/none")}
    put("sep/n_stimuli", sum(v["n"] for v in none.values()))
    put("sep/speech_max", pct(max(v["dbs_speech"] for v in none.values()), 1))
    put("sep/vocal_min", pct(min(v["dbs_speech"] + v["dbs_sfx"] for v in none.values())))
    put("sep/uvr_breath", pct(none["breathes"]["uvr_vocals"]))
    put("sep/uvr_min_other", pct(min(v["uvr_vocals"] for k, v in none.items() if k != "breathes")))
    put("sep/dnsmos_min", f"{-max(v['dnsmos_delta'] for v in none.values()):.2f}")
    put("sep/dnsmos_max", f"{-min(v['dnsmos_delta'] for v in none.values()):.2f}")
    put("sep/cross_min", pct(min(v["dnsmos_cross_3"] for v in none.values())))
    put("sep/cross_max", pct(max(v["dnsmos_cross_3"] for v in none.values())))
    put("sep/speech_lt_half_min", pct(min(v["dbs_speech_lt_half"] for v in none.values())))


def newest(*rels):
    """The first result file that exists: the pipeline as it ships (release model, non-verbal head), else the run
    before it."""
    for r in rels:
        d = J(r)
        if d:
            return d
    return None


def vox():
    compare("vox", newest("compare/voxconverse_rel.json", "compare/voxconverse_ft.json", "compare/voxconverse.json"))
    compare("pil", newest("compare/pilot_nvrel4.json", "compare/pilot_nvrel2.json", "compare/pilot_nvwhisper3.json", "compare/pilot_nvwhisper.json",
                          "compare/pilot_rel.json", "compare/pilot_ft.json"))
    compare("pilb", J("compare/pilot_ft_seg_before_backoff.json"))      # the pilot before the overlap back-off
    compare("pila", J("compare/pilot_ft.json"))      # and after it, same transcriber: the back-off alone
    for k, v in (J("compare/empty_drops.json") or {}).items():      # segments export drops for an empty transcript
        put(f"drops/{k}/share", pct(v["empty"] / v["segments"], 1))     # from the counts: the stored share is rounded
        put(f"drops/{k}/n", f"{v['empty']:,}")
        put(f"drops/{k}/segments", f"{v['segments']:,}")


def compare(tag, d):
    """scripts/compare_emilia.py's output under keys <tag>/<pipeline>/<measure>."""
    if not d:
        return
    put(f"{tag}/recs", d["recordings"])
    put(f"{tag}/hours", f"{d['input_hours']:.2f}")
    for k, p in d["pipelines"].items():
        y, pu = p["yield"], p.get("purity", {})
        put(f"{tag}/{k}/segs", f"{y['segments']:,}")
        put(f"{tag}/{k}/hours", f"{y['hours']:.2f}")
        put(f"{tag}/{k}/kept", pct(y["share_of_input"]))
        put(f"{tag}/{k}/single", pct(pu.get("single_speaker_speech_kept")))
        put(f"{tag}/{k}/impure", pct(pu.get("share_purity_lt_0_9"), 1))
        put(f"{tag}/{k}/second", pct(pu.get("share_second_speaker_ge_0_5s"), 1))
        put(f"{tag}/{k}/overlap", pct(pu.get("share_overlap_ge_0_2s"), 1))
        put(f"{tag}/{k}/lt3", pct(y.get("share_lt_3s")))
        put(f"{tag}/{k}/fullscale", pct(p["audio"]["share_peak_at_full_scale"], 1))
        put(f"{tag}/{k}/overfull", pct(p["audio"]["share_over_full_scale"], 1))
        put(f"{tag}/{k}/kbps", (p["audio"]["bitrate_kbps"] or {}).get("50") and f"{p['audio']['bitrate_kbps']['50']:.0f}")
        put(f"{tag}/{k}/cutoff", (p["audio"].get("cutoff_khz") or {}).get("50") and f"{p['audio']['cutoff_khz']['50']:.1f}")
        put(f"{tag}/{k}/codec", ", ".join(p["audio"].get("codecs") or []))
        cs = p.get("code_switch") or {}
        put(f"{tag}/{k}/en_in_ms", pct(cs.get("english_word_share"), 1))
    for k, e in (d.get("edges_by_alignment") or {}).items():
        put(f"{tag}/{k}/firstcut", pct(e["share_first_word_at_cut"], 1))
        put(f"{tag}/{k}/lastcut", pct(e["share_last_word_at_cut"], 1))
        put(f"{tag}/{k}/eithercut", pct(e["share_either"], 1))
    for k, t in (d.get("throughput") or {}).items():
        if t:
            put(f"{tag}/{k}/xrt", f"{t['x_realtime']:.1f}")
    nv_all = d.get("nonverbal") or {}
    for k, nv in nv_all.items():
        if k in ("recordings", "floor", "source_events") or not isinstance(nv, dict):
            continue
        lg = nv.get("laugh") or {}
        put(f"{tag}/{k}/laugh_kept", pct(lg.get("still_audible")))
        put(f"{tag}/{k}/laugh_tagged", pct(lg.get("tagged")))


from asr_runs import ASR_RUNS, RELEASE, asr as asr_row  # noqa: E402

LANG = {"am": "Amharic", "ar": "Arabic", "az": "Azerbaijani", "bg": "Bulgarian", "bn": "Bengali", "cs": "Czech",
        "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian",
        "fi": "Finnish", "fr": "French", "gu": "Gujarati", "ha": "Hausa", "hi": "Hindi", "hu": "Hungarian",
        "hy": "Armenian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ka": "Georgian", "kk": "Kazakh",
        "km": "Khmer", "kn": "Kannada", "ko": "Korean", "lo": "Lao", "lt": "Lithuanian", "mg": "Malagasy",
        "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "my": "Burmese", "ne": "Nepali",
        "nl": "Dutch", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian",
        "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali",
        "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
        "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese"}


def asr():
    for key in ASR_RUNS:
        s = asr_row(key)
        if not s:
            continue
        for m in ("silence_hr", "music_hr", "nonspeech_hr", "wild_words", "lexicon_recovered"):
            put(f"asr/{key}/{m}", pct(s.get(m), 1))
        # corpus WER (errors over every reference word), the standard one; the per-clip mean is kept in the
        # summary, but on short clips one inserted filler weighs as much as a sentence of errors
        put(f"asr/{key}/librispeech_wer", pct(s.get("librispeech_wer_corpus", s.get("librispeech_wer")), 2))
        put(f"asr/{key}/genuine_wer", pct(s.get("genuine_wer_corpus", s.get("genuine_wer")), 1))
        put(f"asr/{key}/genuine_wer_clipmean", pct(s.get("genuine_wer"), 1))
        put(f"asr/{key}/fleurs", fx(s.get("fleurs_cer_macro")))
        put(f"asr/{key}/cv22", fx(s.get("cv22_cer_macro")))
        put(f"asr/{key}/fleurs_langs", s.get("fleurs_n_langs"))
        put(f"asr/{key}/cv22_langs", s.get("cv22_n_langs"))
        put(f"asr/{key}/runaway", pct(s.get("redup_runaway"), 1))
        put(f"asr/{key}/redup_empty", pct(s.get("redup_empty"), 1))
        e = s.get("empty_on_speech") or {}
        for arm, tag in (("librispeech_test_clean", "ls_empty"), ("genuine", "my_empty"), ("fleurs", "fleurs_empty"),
                         ("cv22", "cv22_empty"), ("speech_in_noise", "noise_empty")):
            put(f"asr/{key}/{tag}", pct(e.get(arm), 1))
        snr = s["fail"].get("noise_empty_by_snr") or {}
        for db, tag in (("20", "noise20"), ("0", "noise0"), ("-5", "noisem5")):
            put(f"asr/{key}/{tag}", pct(snr.get(db), 1))
        langs = s["fail"].get("fleurs_empty_by_lang") or {}
        if langs:
            worst = sorted(langs.items(), key=lambda kv: (-kv[1], kv[0]))
            put(f"asr/{key}/fleurs_empty_max", pct(worst[0][1], 0))
            top = [f"{LANG.get(l, l)} ({100 * v:.0f}\\%)" for l, v in worst[:4] if v > 0]
            if top:
                put(f"asr/{key}/fleurs_empty_top", ", ".join(top[:-1]) + " and " + top[-1] if len(top) > 1 else top[0])
            put(f"asr/{key}/fleurs_empty_langs", sum(v > 0 for v in langs.values()))
        cfg = s.get("config") or {}
        put(f"asr/{key}/clips", cfg.get("n_clips") and f"{cfg['n_clips']:,}")
        put(f"asr/{key}/blank", pct(cfg.get("blank_share")))
        ms = cfg.get("mix_stats") or {}
        put(f"asr/{key}/mix_hours", ms.get("hours") and f"{ms['hours']:.0f}")
        put(f"asr/{key}/mix_langs", ms.get("languages"))
        put(f"asr/{key}/cv22_pool_langs", cfg.get("cv22_pool_languages"))
        put(f"asr/{key}/tts_clips", (ms.get("arms") or {}).get("tts44k") and f"{ms['arms']['tts44k']:,}")
        for arm, tag in (("fleurs", "det"), ("fleurs_forced", "forced")):
            a = s["lid"].get(arm) or {}
            put(f"asr/{key}/{tag}", fx(a.get("macro_cer")))
            put(f"asr/{key}/{tag}_failed", a.get("failed"))
            if a.get("lid_accuracy") is not None:
                put(f"asr/{key}/lid_acc", pct(a["lid_accuracy"], 1))
            if arm == "fleurs":
                put("asr/fleurs_clips", a.get("n") and f"{a['n']:,}")
                put("asr/fleurs_langs", a.get("n_langs"))
    put("asr/release", ASR_RUNS[RELEASE][1].replace("%", "\\%"))
    bl = J("asr/blank_langs.json") or {}
    if bl.get("n"):
        put("asr/blank_n", f"{bl['n']:,}")
        put("asr/blank_en", pct(bl["counts"].get("en", 0) / bl["n"]))
    # vLLM itself against our decoding of the same clips (asr/bench/vllm_check.py), float16 and bfloat16
    vc = J("asr/vllm_check.json") or {}
    blank = [v for k, d in vc.items() if not k.split("/")[0].endswith("_fleurs") for v in d.values()]
    if blank:
        put("asr/vllm/blank_n", f"{sum(v['n'] for v in blank) // max(1, len([k for k in vc if not k.split('/')[0].endswith('_fleurs')])):,}")
        put("asr/vllm/blank_same", pct(min(v.get("same_transcript", 0) for v in blank), 1))
    for k, d in vc.items():
        f = d.get("fleurs")
        if f:
            tag = "bf16" if k.endswith("bfloat16") else "fp16"
            put(f"asr/vllm/fleurs_{tag}_same", pct(f.get("same_transcript"), 1))
            put(f"asr/vllm/fleurs_{tag}_lang", pct(f.get("same_language"), 1))
            put(f"asr/vllm/fleurs_{tag}_cer", fx(f.get("cer_macro_vllm")))
            put("asr/vllm/fleurs_hf_cer", fx(f.get("cer_macro_hf")))
            put("asr/vllm/fleurs_n", f"{f['n']:,}")
    # the same checkpoints decoded twice: Hugging Face's default suppression against vLLM's none
    sup = J("asr/suppress_check.json") or {}
    for m, modes in sup.items():
        for mode, arms in modes.items():
            n = sum(a["n"] for a in arms.values())
            if len(arms) == 3 and n:
                put(f"asr/sup/{m}/{mode}", pct(sum(a["words_rate"] * a["n"] for a in arms.values()) / n, 1))
                put(f"asr/sup/{m}/{mode}_any", pct(sum(a["any_output_rate"] * a["n"] for a in arms.values()) / n, 0))
                put("asr/sup/n", f"{n:,}")
            for arm, a in arms.items():
                put(f"asr/sup/{m}/{mode}/{arm}", pct(a["words_rate"], 1))
                put(f"asr/sup/{arm}_n", f"{a['n']:,}")


EMO_HEADS = ("whisper", "whisper_ft", "w2vbert", "wavlm", "e2v")
ENSEMBLE = "whisper + whisper_ft + w2vbert + wavlm + e2v + meralion + e2v"


def emotion_mer2023():
    """MER2023 (Mandarin film and TV) trained in: each head left out of it (LOCO), and the held-out average with it
    beside the released one (deploy/box_emo_mer2023.sh; scripts/compare_emotion_ensembles.py)."""
    for h in EMO_HEADS:
        r = J(f"emotion/mer2023/probe_{h}.json") or {}
        lo = (r.get("loco") or {}).get("mer2023")
        if lo:
            put("emo3/mer/n", f"{lo['n']:,}")
            put(f"emo3/mer/loco/{h}", fx(lo["uar"], 2))
            put(f"emo3/mer/fear/{h}", pct(lo["recall"].get("fearful")))
            put(f"emo3/h_{h}/uar", fx(r["held_out"]["all"]["uar"]))
    ens = (J("emotion/mer2023/ensemble_all.json") or {}).get("results", {})
    seven = [k for k in ens if k.count("+") == 6]
    if seven:
        put("emo3/ens/uar", fx(ens[seven[0]]["held_out"]["uar"]))
    c = J("emotion/mer2023/compare_per_corpus.json") or {}
    for key, name in (("emo2c", "live"), ("emo3", "with_mer2023")):
        for corpus, tag in (("cameo", "nemo"), ("nvtts", "nvtts")):
            r = (c.get(name) or {}).get(corpus)
            if r:
                put(f"{key}/{tag}", fx(r["uar"]))


def emotion_heads():
    """Melati's own heads (emotion/probe.py), the released classifiers on the same clips (emotion/score_current.py),
    and the seven averaged (emotion/ensemble.py)."""
    ms = J("emotion/manifest_stats.json") or {}
    if ms:
        put("emo2/clips", f"{ms['clips']:,}")
        put("emo2/train_clips", f"{ms['train']['clips']:,}")
        put("emo2/train_corpora", ms["train"]["corpora"])
        put("emo2/train_langs", len(ms["train"]["languages"]))
        put("emo2/held_clips", f"{ms['held_out']['clips']:,}")
        for c in ("meld", "expresso", "emovdb"):
            if c in ms["per_corpus"]:
                put(f"emo2/{c}_n", f"{ms['per_corpus'][c]['n']:,}")
    mt = J("emotion/my_test_sample.json") or {}
    put("emo/my_sample", mt.get("clips") and f"{mt['clips']:,}")
    put("emo/my_random", mt.get("random") and f"{mt['random']:,}")
    cur = J("emotion/current.json") or {}
    for m, r in cur.items():
        k = m.replace("+", "_")
        put(f"emo2/{k}/uar", fx(r["held_out"]["all"]["uar"]))
        put(f"emo2/{k}/acc", pct(r["held_out"]["all"]["acc"]))
    for h in EMO_HEADS:
        r = J(f"emotion/probe_{h}.json") or {}
        if r:
            put(f"emo2/h_{h}/uar", fx(r["held_out"]["all"]["uar"]))
            put(f"emo2/h_{h}/acc", pct(r["held_out"]["all"]["acc"]))
            put(f"emo2/h_{h}/loco", fx(r.get("loco_mean_uar")))
            put(f"emo2/h_{h}/nemo", fx(r["held_out"]["nemo"]["uar"]))
    ens = (J("emotion/ensemble_all.json") or {}).get("results", {})
    for name, key in ((ENSEMBLE, "ens"), ("meralion + e2v", "cur")):
        r = ens.get(name)
        if not r:
            continue
        put(f"emo2/{key}/uar", fx(r["held_out"]["uar"]))
        put(f"emo2/{key}/acc", pct(r["held_out"]["acc"]))
        for lab, v in (r.get("thresholds_cv") or {}).items():
            put(f"emo2/{key}/{lab}/recall", pct(v["recall"]) if v else "none")
            put(f"emo2/{key}/{lab}/prec", pct(v["precision_at_prior"]) if v else None)
        for lab, v in (r.get("thresholds") or {}).items():
            put(f"emo2/{key}/{lab}/t", v and f"{v['threshold']:.2f}")


def fillers():
    """PodcastFillers test episodes through Melati, with base Whisper and with the released model (scripts/eval_fillers.py)."""
    for tag in ("base", "rel"):
        d = J(f"fillers/podcastfillers_{tag}.json")
        if not d:
            continue
        put("fill/episodes", d.get("episodes"))
        put("fill/tol_ms", d.get("tol_s") and f"{1000 * d['tol_s']:.0f}")
        for scope, block in (d.get("scopes") or {}).items():
            for cls, m in block.items():
                put(f"fill/{tag}/{scope}/{cls}/p", pct(m.get("precision"), 1))
                put(f"fill/{tag}/{scope}/{cls}/r", pct(m.get("recall"), 1))
                put(f"fill/{tag}/{scope}/{cls}/pred", m.get("pred") is not None and f"{m['pred']:,}")
                put(f"fill/{scope}/{cls}/ref", m.get("ref") is not None and f"{m['ref']:,}")


def emotion():
    d = J("emotion/eval.json")
    if not d:
        return
    put("emo/n", f"{d['n']:,}")
    put("emo/corpora", len(d["corpora"]))
    put("emo/seen", len(d["seen"].get("e2v", [])))
    held = [c for c in d["corpora"] if c not in set(d["seen"].get("e2v", []))]
    put("emo/held_n", f"{sum(d['corpora'][c] for c in held):,}")
    put("emo/target", pct(d["target_precision"]))
    for k, m in d["models"].items():
        kk = k.replace("+", "_")
        put(f"emo/{kk}/acc", fx(m["acc"], 2)); put(f"emo/{kk}/uar", fx(m["uar"], 2))
        put(f"emo/{kk}/held_acc", fx(m["held_out"]["acc"], 2)); put(f"emo/{kk}/held_uar", fx(m["held_out"]["uar"], 2))
        for c in ("jl_corpus", "pavoque", "nemo", "nonverbaltts"):
            put(f"emo/{kk}/{c}", fx(m["per_corpus"].get(c, {}).get("acc"), 2))
        for lab, t in m["thresholds"].items():
            if t:
                put(f"emo/{kk}/{lab}/t", fx(t["threshold"], 2))
                put(f"emo/{kk}/{lab}/p_prior", fx(t["precision_at_prior"], 2))
                put(f"emo/{kk}/{lab}/p_bal", fx(t["precision_balanced"], 2))
                put(f"emo/{kk}/{lab}/recall", fx(t["recall"], 2))
    for lab, v in d["priors"].items():
        put(f"emo/prior/{lab}", pct(v))


NV_ENCODERS = ("whisper", "wavlm", "ast")
NV_RULE = '{"floor_db": 2}'             # the candidate rule the pipeline ships (melati.nv_gaps FLOOR_DB)
NV_TAGGED = ("breath", "laugh", "cough")    # the classes test can measure, so the only ones tagged


def nv_heads():
    """Melati's own non-verbal heads (nonverbal/): what they trained on, how each scores on clips nothing trained
    on, the gap protocol beside Qwen3-Omni's (same candidates, same calibration, same scoring), PodcastFillers
    through the pipeline, and the AudioSet audit of the tags PodcastFillers has no label for."""
    ms = J("nonverbal/manifest_stats.json") or {}
    for split in ("train", "validation", "test"):
        for corpus in ("podcastfillers", "vocalsound"):
            block = ms.get(f"{corpus}/{split}") or {}
            put(f"nvh/n/{corpus}/{split}", block and f"{sum(block.values()):,}")
    h = J("nonverbal/head_whisper.json") or {}
    by = {}
    for k, v in (h.get("train") or {}).items():
        by[k.split("/")[0]] = by.get(k.split("/")[0], 0) + v
    for corpus, key in (("pf_cands", "pfc"), ("nvtts_cands", "nvc"), ("podcastfillers", "pf"), ("vocalsound", "vs")):
        put(f"nvh/n/{key}", by.get(corpus) and f"{by[corpus]:,}")
    put("nvh/n/train", by and f"{sum(by.values()):,}")
    for tag, suffix in (("nvh", ""), ("nvhe", "_events")):
        for enc in NV_ENCODERS:
            r = (J(f"nonverbal/head_{enc}{suffix}.json") or {}).get("test", {}).get("all")
            if r:
                put(f"{tag}/{enc}/f1", fx(r["macro_f1"], 2))
                for lab, m in r["per_class"].items():
                    put(f"{tag}/{enc}/{lab.replace(' ', '_')}/f1", fx(m["f1"], 2))
        g = J(f"nonverbal/nonverbaltts_gaps_heads{suffix}.json")
        if g:
            put(f"{tag}/cands", f"{g['candidates']:,}")
            for name, v in g["variants"].items():
                key = "mix" if "+" in name else name
                for c, t in v["thresholds"].items():
                    if t:
                        put(f"{tag}/thr/{key}/{c}", fx(t["threshold"], 2))
                for c, r in v["calibrated"].items():
                    put(f"{tag}/gap/{key}/{c}/tagp", pct(r.get("tag_precision")))
                    put(f"{tag}/gap/{key}/{c}/tagr", pct(r.get("tag_recall")))
    # the gap finder: why test breaths had no candidate under the old rule and the new one (nonverbal/gap_rule.py)
    for key in ("old", "new"):
        d = J(f"nonverbal/gap_diagnose_test_{key}.json") or {}
        why = (d.get("breath") or {}).get("why") or {}
        n = sum(why.values())
        if n:
            for k, tag in (("found", "found"), ("no gap", "nogap"), ("too brief", "brief"), ("too quiet", "quiet")):
                put(f"gap/{key}/{tag}", pct(why.get(k, 0) / n))
                put(f"gap/{key}/{tag}/n", f"{why.get(k, 0):,}")
            q = d["breath"].get("quiet_margin_db") or []
            put(f"gap/{key}/quiet_db", q and fx(float(np.median(q)), 1))
    sw = J("nonverbal/gap_sweep_test.json") or {}
    for key, rule in (("old", "{}"), ("new", NV_RULE)):        # "{}" ran before the defaults changed: the old rule
        if rule in sw:
            put(f"gap/{key}/per_clip", fx(sw[rule]["per_clip"], 1))
            put(f"gap/{key}/candr", pct(sw[rule]["candidate_recall"].get("breath")))
    for key, rel in (("nv3", "gap_eval_v3head"), ("nvf", "gap_eval_rel2")):
        # nv3: the new rule with the head retrained on its candidates (VocalSound in); nvf: the released head, which
        # the pipeline ships (PodcastFillers' CC BY episodes, NonverbalTTS, COUGHVID)
        ev = (J(f"nonverbal/{rel}.json") or {}).get(NV_RULE)
        if not ev:
            continue
        for c, r in ev["test"].items():
            put(f"{key}/{c}/tagp", pct(r.get("tag_precision")))
            put(f"{key}/{c}/tagr", pct(r.get("tag_recall")))
        for c in NV_TAGGED:
            t = ev["thresholds"].get(c)
            put(f"{key}/thr/{c}", t and fx(t["threshold"], 2))
    final = (J("nonverbal/gap_eval_rel2.json") or {}).get(NV_RULE)
    omni = (J("nonverbal/nonverbaltts_gaps.json") or {}).get("test", {}).get("calibrated", {}).get("laugh", {})
    head = (final or {}).get("test", {}).get("laugh", {})
    if omni.get("tag_recall") and head.get("tag_recall"):
        put("nvh/laugh_x", fx(head["tag_recall"] / omni["tag_recall"], 1))      # laughs found, head over Qwen3-Omni
    for key, rel in (("nvw", f"podcastfillers_nv{NV_SHIP}"), ("nvp", "podcastfillers_nvrel2"),
                     ("nvg", "podcastfillers_nvrel4pf"), ("nvw3", "podcastfillers_nvwhisper3"),
                     ("nvw2", "podcastfillers_nvwhisper"),
                     ("nvh", "podcastfillers_nvheads"),
                     ("nvhe", "podcastfillers_nvheads_events"), ("rel", "podcastfillers_rel")):
        pf = J(f"fillers/{rel}.json")
        if not pf:
            continue
        un = pf.get("unmatched_predictions_overlap") or {}
        for scope, block in (pf.get("scopes") or {}).items():
            for cls, m in block.items():
                put(f"fill/{key}/{scope}/{cls}/p", pct(m.get("precision"), 1))
                put(f"fill/{key}/{scope}/{cls}/r", pct(m.get("recall"), 1))
                put(f"fill/{key}/{scope}/{cls}/pred", m.get("pred") is not None and f"{m['pred']:,}")
        for cls in ("laughter", "breath"):
            m = (pf.get("scopes") or {}).get("inside", {}).get(cls) or {}
            if not m.get("pred"):
                continue
            hit = round((m.get("precision") or 0) * m["pred"])
            wrong = sum(v for k, v in (un.get(cls) or {}).items() if k != "(no label)")
            put(f"fill/{key}/lab/{cls}/p", pct(hit / max(hit + wrong, 1), 0))       # where PodcastFillers has a label
            put(f"fill/{key}/unl/{cls}", f"{(un.get(cls) or {}).get('(no label)', 0):,}")
    ev = J("compare/pilot_events.json") or {}             # tags in the 200-podcast pilot's export, per labeller
    for key, tag in (("qwen3_omni", "qwen"), ("head_old_rule", "old"), ("head_vocalsound", "vs"), ("head_rel2", "prev"),
                     ("head", "head")):
        r = ev.get(key)
        if r:
            for kind, v in r["events"].items():
                put(f"pev/{tag}/{kind}", f"{v:,}")
                put(f"pev/{tag}/{kind}/h", fx(r["per_hour"][kind], 0 if r["per_hour"][kind] >= 10 else 1))
            put("pev/hours", fx(r["hours"], 1))
    a = J(f"fillers/audit_nv_tags_{NV_SHIP}.json") or J("fillers/audit_nv_tags_rel2.json") or {}
    for key, grp in (("unl", "breath/unlabelled"), ("hand", "breath/control"), ("matched", "breath/matched"),
                     ("words", "words/control")):
        r = a.get(grp)
        if r:
            put(f"audit/breath/{key}", pct(r["breath_like"]))
            put(f"audit/breath/{key}/n", f"{r['n']:,}")
            put(f"audit/breath/{key}/speech", fx(r["mean_Speech"], 2))
    for key, grp in (("unl", "laughter/unlabelled"), ("hand", "laughter/control")):
        for tag, src in (("ship", a), ("gap", J("fillers/audit_nv_tags_rel4pf.json") or {}),
                         ("prev", J("fillers/audit_nv_tags_rel2.json") or {})):
            r = src.get(grp)
            if r:
                put(f"audit/laugh/{tag}/{key}", fx(r["mean_Laughter"], 3))
                put(f"audit/laugh/{tag}/{key}/n", f"{r['n']:,}")
                put(f"audit/laugh/{tag}/{key}/gasp", pct(r["top1"].get("Gasp", 0) / r["n"]))


NV_SHIP = "rel4f"            # the pipeline run with the released model: gap head rel4 (breaths), frame head v3


def nv_release():
    """The released non-verbal model: why NonverbalTTS left its training, what calibrating on PodcastFillers instead
    gives, and the frame head (nonverbal/frames.py) for laughs and coughs over speech, on every test set."""
    ev = J("nonverbal/gap_eval_rel2nonv.json") or {}
    r = (ev.get(NV_RULE) or {})
    for c in ("breath", "laugh"):
        t = (r.get("thresholds") or {}).get(c)
        put(f"nonv/thr/{c}", t and fx(t["threshold"], 2))
        put(f"nonv/nv/{c}/tagr", pct((r.get("test") or {}).get(c, {}).get("tag_recall")))
    for c in ("breath", "laugh", "cough"):                       # the previous release, which trained on NonverbalTTS
        x = ((J("nonverbal/gap_eval_rel2.json") or {}).get(NV_RULE) or {}).get("test", {}).get(c) or {}
        put(f"prev/nv/{c}/tagr", pct(x.get("tag_recall"))); put(f"prev/nv/{c}/tagp", pct(x.get("tag_precision")))
    cal = (J("nonverbal/gap_eval_rel4pf.json") or {}).get(NV_RULE, {}).get("thresholds") or {}
    for c in ("breath", "laugh"):
        if cal.get(c):
            put(f"rel/thr/{c}", fx(cal[c]["threshold"], 2))
            put(f"rel/cal/{c}/p", pct(cal[c]["precision"])); put(f"rel/cal/{c}/r", pct(cal[c]["recall"]))
    cv = (J("nonverbal/gap_curve_rel4.json") or {}).get("curves", {}).get("test", {})
    if cal.get("breath") and cv:
        x = cv["breath"][str(cal["breath"]["threshold"])]
        put("rel/nv/breath/tagr", pct(x["tag_recall"])); put("rel/nv/breath/tagp", pct(x["tag_precision"]))
        x = cv["laugh"][str(cal["laugh"]["threshold"])]
        put("rel/nv/laugh/tagr", pct(x["tag_recall"])); put("rel/nv/laugh/tagp", pct(x["tag_precision"]))
    h = J("nonverbal/head_whisper_rel4.json") or {}
    by = {}
    for k, v in (h.get("train") or {}).items():
        by[k.split("/")[0]] = by.get(k.split("/")[0], 0) + v
    for corpus, key in (("ami_cands", "amic"), ("ami_unmarked", "amiu"), ("coswara", "coswara"),
                        ("coughvid", "coughvid")):
        put(f"rel/n/{key}", by.get(corpus) and f"{by[corpus]:,}")
    put("rel/n/train", by and f"{sum(by.values()):,}")
    # AMI's test meetings: where its laughs and coughs are, and what each model finds
    for tag, f in (("prev", "ami_eval_rel2"), ("gap", "ami_eval_rel4pf")):
        a = (J(f"nonverbal/{f}.json") or {}).get("test", {}).get("nonverbaltts_thresholds") or {}
        for c in ("laugh", "cough"):
            x = a.get(c) or {}
            if x:
                put(f"ami/{tag}/{c}/r", pct(x.get("recall"))); put(f"ami/{tag}/{c}/p", pct(x.get("precision")))
                put(f"ami/{c}/n", f"{x['marks']:,}")
                put(f"ami/{c}/nocand_speech", pct(x["no_candidate_in_speech"] / x["marks"]))
                put(f"ami/{c}/with_cand", pct(x["with_candidate"] / x["marks"]))
    fe = J("nonverbal/frames3_eval_pf.json") or {}
    if fe:
        lt = (fe.get("podcastfillers") or {}).get("threshold")
        ct = fe["ami"]["cough"]["threshold"]
        put("frm/thr/laugh", lt and fx(lt, 2)); put("frm/thr/cough", ct and fx(ct, 3))
        al = fe["ami"]["laugh"].get("test_at_pf_threshold") or fe["ami"]["laugh"]["test"]
        ac = fe["ami"]["cough"]["test"]
        for c, x in (("laugh", al), ("cough", ac)):
            put(f"frm/ami/{c}/r", pct(x["recall"])); put(f"frm/ami/{c}/p", pct(x["precision"]))
            put(f"frm/ami/{c}/rs", pct(x["recall_in_speech"]))
        pf = (fe.get("podcastfillers") or {}).get("test") or {}
        put("frm/pf/laugh/r", pct(pf.get("recall"))); put("frm/pf/laugh/p", pct(pf.get("precision")))
        put("frm/pf/laugh/n", pf.get("refs") and f"{pf['refs']:,}")
        put("frm/pf/laugh/cal_p", pct(((fe.get("podcastfillers") or {}).get("validation") or {}).get("precision")))
    sw = J("nonverbal/frames_sweep_pf.json")       # laughs at every threshold, through the pipeline (sweep_frames.py)
    if sw and fe:
        ta, tp = fe["ami"]["laugh"]["threshold"], (fe.get("podcastfillers") or {}).get("threshold")
        put("frm/ds/thr", fx(ta, 2))
        for k, x in (("ds", sw["laugh"].get(str(ta))), ("ship", sw["laugh"].get(str(tp))),
                     ("prev", sw["others"]["previous release"]), ("omni", sw["others"]["Qwen3-Omni"])):
            if x:
                put(f"frm/sweep/{k}/r", pct(x["recall"])); put(f"frm/sweep/{k}/p", pct(x["precision_labelled"]))
                put(f"frm/sweep/{k}/r1", pct(x["recall"], 1))
    fn = (J("nonverbal/frames3_nvtts.json") or {}).get("test") or {}
    for c, t in (("laugh", fe and (fe.get("podcastfillers") or {}).get("threshold")),
                 ("cough", fe and fe["ami"]["cough"]["threshold"])):
        x = (fn.get(c) or {}).get(str(t)) if t else None
        if x:
            put(f"frm/nv/{c}/r", pct(x["recall"])); put(f"frm/nv/{c}/p", pct(x["precision"]))
            put(f"frm/nv/{c}/hyp", x["hyp"])
    v4 = J("nonverbal/frames4_eval_pf.json") or {}
    if v4:                                                     # the phone-recorded coughs added: they did not help
        put("frm4/ami/cough/r", pct(v4["ami"]["cough"]["test"]["recall"]))
        put("frm4/ami/cough/p", pct(v4["ami"]["cough"]["test"]["precision"]))
    d = J("nonverbal/gap_diagnose_test_new.json") or {}
    why = (d.get("laugh") or {}).get("why") or {}
    if why:
        put("gap/new/laugh/nogap/n", why.get("no gap", 0)); put("gap/new/laugh/n", sum(why.values()))


def nonverbal():
    g = J("nonverbal/nonverbaltts_gaps.json")
    if g:
        put("nv/test_clips", g["test"]["clips"])
        put("nv/calib_clips", g["calibration"]["clips"])
        put("nv/target", pct(g["calibration"]["target_precision"]))
        put("nv/test_cands", sum(g["test"]["labels"].values()))
        for c, r in g["test"]["calibrated"].items():
            put(f"nv/gap/{c}/tagp", pct(r.get("tag_precision"))); put(f"nv/gap/{c}/tagr", pct(r.get("tag_recall")))
            put(f"nv/gap/{c}/candr", pct(r.get("candidate_recall")))
            put(f"nv/gap/{c}/ref", r.get("ref"))
        for c, r in g["test"]["argmax"].items():
            put(f"nv/gaparg/{c}/tagp", pct(r.get("tag_precision"))); put(f"nv/gaparg/{c}/tagr", pct(r.get("tag_recall")))
            put(f"nv/gaparg/{c}/candr", pct(r.get("candidate_recall")))
        for c, t in g["calibration"]["thresholds"].items():
            if t:
                put(f"nv/thr/{c}", fx(t["threshold"], 2))
    nv_heads()
    nv_release()
    for key, tag in (("nonverbaltts_omni", "omni"), ("nonverbaltts_omni_breath", "omnib"), ("nonverbaltts_omni_strict", "omnis")):
        d = J(f"nonverbal/{key}.json")
        if not d:
            continue
        for c, r in d["per_class"].items():
            put(f"nv/{tag}/{c}/p", pct(r["precision"])); put(f"nv/{tag}/{c}/r", pct(r["recall"]))
            put(f"nv/{tag}/{c}/place", pct(r.get("placement")))
            put(f"nv/{tag}/{c}/fp", r.get("fp"))
    panns = J("nonverbal/nonverbaltts_panns.json")
    if panns:
        rt = panns["reference_tags"]
        for c, n in rt.items():
            put(f"nv/ref/{c}", n)
        best = {}
        for f, cls in panns["by_floor"].items():
            for c, r in cls.items():
                f1 = 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9)
                if c not in best or f1 > best[c][0]:
                    best[c] = (f1, r)
        for c, (_, r) in best.items():
            put(f"nv/panns/{c}/p", pct(r["precision"])); put(f"nv/panns/{c}/r", pct(r["recall"], 1))


def dataset():
    """Melati v1 as published (scripts/build_hf_dataset.py's stats), the full run's throughput
    (scripts/full_timeline.py) and the batched encoders' check (scripts/check_ssl_batching.py)."""
    d = J("dataset/stats.json")
    if d:
        put("ds/recs", f"{d['release_recordings']:,}"); put("ds/src_recs", f"{d['source_recordings']:,}")
        put("ds/complete", "yes" if d.get("complete") else "no"); put("ds/blocks", d["blocks"])
        put("ds/src_h", f"{d['release_source_hours']:,.0f}"); put("ds/hours", f"{d['hours']:,.0f}")
        put("ds/kept", pct(d["hours"] / d["release_source_hours"]))
        put("ds/segments", f"{d['segments']:,}"); put("ds/speakers", f"{d['speakers']:,}")
        if "windowed" in d:
            put("ds/win/recs", f"{d['windowed']['recordings']:,}"); put("ds/win/h", f"{d['windowed']['hours']:,.0f}")
        put("ds/empty", d["empty_source_files"]); put("ds/failed", len(d["failed_recordings"]))
        put("ds/gib", f"{d['parquet_gib']:,.0f}"); put("ds/shards", d["shards"])
        for k, h in d["hours_by_language"].items():
            put(f"ds/lang/{k}/h", f"{h:,.0f}" if h >= 10 else f"{h:.1f}")
            put(f"ds/lang/{k}/share", pct(h / d["hours"], 1))
        for k, v in d["events_per_hour"].items():
            put(f"ds/ev/{k.split()[-1]}/ph", f"{v:.1f}" if v >= 1 else f"{v:.2f}")     # "clears throat" -> throat
        for k, v in (d.get("delivery_per_hour") or {}).items():
            put(f"ds/dlv/{k}/ph", f"{v:.2f}")
        put("ds/fill/ph", f"{d['fillers_per_hour']:.0f}")
        for k, v in d["emotion_spans"].items():
            put(f"ds/emo/{k}/ph", f"{v / d['hours']:.1f}")
        put("ds/laughs_over", pct(d["laughs_over_speech"] / max(d["events"].get("laughs", 0), 1)))
        put("ds/emilia/h", f"{d['emilia_filter']['hours']:,.0f}"); put("ds/emilia/segs", f"{d['emilia_filter']['segments']:,}")
        put("ds/emilia/share", pct(d["emilia_filter"]["hours"] / d["hours"]))
        put("ds/mos50", f"{d['mos']['50']:.2f}"); put("ds/dnsmos50", f"{d['dnsmos']['50']:.2f}")
        put("ds/dur50", f"{d['duration_s']['50']:.1f}")
        put("ds/vocal", pct(d["audio_version"].get("vocal", 0) / d["segments"]))
        lo = d.get("left_out") or {}               # rows the export leaves out (melati.stage_export.release_drop)
        if lo:
            n, h = sum(v["segments"] for v in lo.values()), sum(v["hours"] for v in lo.values())
            put("ds/left/n", f"{n:,}"); put("ds/left/h", f"{h:.1f}"); put("ds/left/share", pct(n / (n + d["segments"]), 1))
            for k, v in lo.items():
                put(f"ds/left/{k}/n", f"{v['segments']:,}")
        if "squeezed" in d:
            put("ds/squeezed", pct(d["squeezed"]["segments"] / d["segments"], 1))
        snd, sn = d.get("sound_tags") or {}, d.get("sounds") or {}
        if snd and sn:                             # the sounds stage: its point tags and the `sounds` column
            put("ds/snd/music/ph", f"{d['sound_tags_per_hour'].get('music', 0):.1f}")
            others = sorted(((k, v) for k, v in snd.items() if k != "music"), key=lambda kv: -kv[1])
            put("ds/snd/other/n", f"{sum(v for _, v in others):,}")
            put("ds/snd/other/kinds", ", ".join(f"{k} {v:,}" for k, v in others[:4])
                + (f", and {len(others) - 4} more kinds" if len(others) > 4 else ""))
            put("ds/snd/rows", f"{d['rows_with_sound_tags']:,}")
            put("ds/snd/music_share", pct(sn["music_rows"] / d["segments"]))
            put("ds/snd/music_mix_share", pct(sn["music_by_audio_version"].get("mix", 0) / max(sn["music_rows"], 1)))
    t = J("dataset/timeline.json")
    if t:
        put("ds/run/wall_h", f"{t['wall_h']:.0f}"); put("ds/run/xrt", f"{t['x_realtime']:.0f}")
        put("ds/run/recs", f"{t['recordings']:,}"); put("ds/run/audio_h", f"{t['audio_h']:,.0f}")
        put("ds/run/peak_xrt", f"{max(p['x_realtime'] or 0 for p in t['points']):.0f}")
    s = J("dataset/ssl_batching.json")
    if s:
        e = s["encoders"]
        put("ds/ssl/clips", s["clips"]); put("ds/ssl/w2v", f"{e['w2vbert']['speedup']:.1f}")
        put("ds/ssl/wavlm", f"{e['wavlm']['speedup']:.1f}")
        put("ds/ssl/cos", f"{min(v['feature_min_cosine'] for v in e.values()):.4f}")
        put("ds/ssl/p", f"{max(v['prob_max_abs_diff'] for v in e.values()):.3f}")


def write_numbers():
    lines = ["% generated by make_tables.py from results/; do not edit", r"\makeatletter",
             r"\newcommand{\val}[1]{\ifcsname v@#1\endcsname\csname v@#1\endcsname\else\textbf{??}"
             r"\PackageWarning{val}{undefined value #1}\fi}"]
    for k in sorted(NUM):
        lines.append(r"\expandafter\def\csname v@" + k + r"\endcsname{" + str(NUM[k]) + "}")
    lines.append(r"\makeatother")
    # Melati v1's section is worded for the run under way until every shard is published (stats.json's complete)
    lines += [r"\newif\ifdscomplete"] + ([r"\dscompletetrue"] if NUM.get("ds/complete") == "yes" else [])
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"-> preprint/numbers.tex ({len(NUM)} values)")


ABSTRACT_MAX = 1900


def abstract_txt():
    tex = (HERE / "neurips_2023.tex")
    if not tex.exists():
        return
    t = tex.read_text()
    a = t[t.index(r"\begin{abstract}") + len(r"\begin{abstract}"):t.index(r"\end{abstract}")]
    a = re.sub(r"\\val\{([^}]+)\}", lambda m: str(NUM.get(m.group(1), "??")).replace("\\%", "%"), a)
    a = re.sub(r"\\url\{([^}]+)\}", r"\1", a)
    a = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", a).replace("\\%", "%").replace("~", " ")
    a = re.sub(r"\s+", " ", a).strip()
    (HERE / "abstract.txt").write_text(a + "\n")
    print(f"-> preprint/abstract.txt ({len(a)} characters)")
    if len(a) > ABSTRACT_MAX:
        raise SystemExit(f"abstract is {len(a)} characters, over {ABSTRACT_MAX}")


def audioset():
    """AudioSet's taggers (nonverbal/audioset_eval.py: PretrainedSED's BEATs, ATST-F and M2D averaged) at the export's
    thresholds (export_config.json), each half of NonverbalTTS's clips measured at the threshold the other chose, and
    their laughs beside the frame head's through the pipeline on PodcastFillers (nonverbal/sweep_frames.py)."""
    d, cfg = J("nonverbal/audioset_BEATs+ATST-F+M2D.json"), J("dataset/export_config.json") or {}
    if not d:
        return
    rng = lambda xs: (pct(min(xs)) if round(100 * min(xs)) == round(100 * max(xs))  # noqa: E731
                      else f"{100 * min(xs):.0f} to {pct(max(xs))}")
    names = {"sniff": "sniffs", "throat_clear": "clears throat", "cough": "coughs", "laugh": "laughs"}
    for cls, t in (cfg.get("audioset_thresholds") or {}).items():
        h = d["nvtts"].get(names.get(cls, cls) + "/halves")
        if h:
            ms = [x[str(t)] for x in h]
            put(f"as/{cls}/t", fx(t, 2)); put(f"as/{cls}/p", rng([m["precision"] for m in ms]))
            put(f"as/{cls}/r", rng([m["recall"] for m in ms]))
    put("as/nvtts/h", f"{d['nvtts']['hours']:.1f}"); put("as/nvtts/clips", f"{d['nvtts']['clips']:,}")
    w = (cfg.get("audioset_delivery") or {}).get("whisper")
    cov = ((d.get("expresso") or {}).get("covers_half") or {}).get("whisper", {}).get(str(w)) if w else None
    if cov:
        put("as/whisper/t", fx(w, 1)); put("as/whisper/found", pct(cov["whisper"]))
        put("as/whisper/others", pct(max(v for k, v in cov.items() if k != "whisper")))
        put("as/whisper/n_others", f"{sum(n for k, n in d['expresso']['utterances'].items() if k != 'whisper'):,}")
    sw, u = J("nonverbal/frames_sweep_pf_audioset.json"), (cfg.get("audioset_thresholds") or {}).get("laugh")
    if sw and u:
        tf = cfg["nv_frame_thresholds"]["laugh"]
        x, alone = sw["laugh"].get(f"{tf}+audioset_{u}"), sw["laugh"].get(str(tf))
        if x and alone:
            put("as/union/r", pct(x["recall"])); put("as/union/p", pct(x["precision_labelled"]))
            put("as/union/ph", f"{sw['tags_per_hour'][f'{tf}+audioset_{u}']['laughs']:.1f}")
            put("as/frame/ph", f"{sw['tags_per_hour'][str(tf)]['laughs']:.1f}")


AS_RESULTS = "audioset/student_v5_release_fp32.json"
AS_STAGES = (("train_clean_e10", "clean_e10/best.pt"), ("train_big_from_v1", "big_from_v1/best.pt"),
             ("train_frames1_fw0", "frames1_fw0/clip.pt"), ("train_v5_big6", "v5_big6/best.pt"))
AS_FRAMES = "frames1/clip.pt"            # stage 3 trained with the frame head beside it: v3
AS_M = "melati:/mnt/data/melati/models/audioset/"      # results keys: where the box kept each checkpoint, not a path
AS_THREE = "ced-base + eat-large + sslam"                    # stages 1 to 3's teachers, averaged
AS_FIVE = f"ced-base + eat-large + {AS_M}frames1/clip.pt + {AS_M}frames1_fw0/clip.pt + sslam"
AS_SIX = f"beats + {AS_FIVE}"                                # stage 4's teachers: the released ensemble's members


def as_tagger():
    """Melati's own AudioSet tagger (audioset/): the released taggers and each stage of the student on the same
    evaluation clips in fp32 (audioset/evaluate.py), what bf16 costs (the taggers scored both ways), each stage's pool
    and settings (its training log), and its frame head against PretrainedSED's on the strongly labelled test clips
    (audioset/frames_eval.py)."""
    r, b = J(AS_RESULTS), J("audioset/teachers_eval_bf16.json")
    if not r:
        return
    m = r["models"]
    put("astag/clips", f"{r['clips']:,}")
    for k in ("ced-base", "eat-large", "sslam", "eat-base", "ast", "beats"):
        put(f"astag/m/{k}", fx(m[k], 4))
    put("astag/avg", fx(r["averages"][AS_THREE], 4))
    put("astag/avg6", fx(r["averages"][AS_SIX], 4))
    for n, (log, ckpt) in enumerate(AS_STAGES, 1):
        put(f"astag/s{n}", fx(m.get(f"melati:/mnt/data/melati/models/audioset/{ckpt}"), 4))
        rec = [json.loads(line) for line in (RES / "audioset" / f"{log}.jsonl").read_text().splitlines()]
        a = next(x["args"] for x in rec if "args" in x)
        put(f"astag/s{n}/epochs", f"{float(a['epochs']):g}")
    if b:
        put("astag/bf16gap", fx(100 * max(abs(b["models"][k] - m[k]) for k in b["models"]), 1))
    for name in ("clean", "big", "big6"):
        pool = J(f"audioset/pool_{name}.json")
        if pool:
            put(f"astag/pool/{name}/clips", f"{pool['clips']:,}")
            put(f"astag/pool/{name}/h", f"{pool['hours']:,.0f}")
            put(f"astag/pool/{name}/strong", f"{pool['with_timed_events']:,}")
            put(f"astag/pool/{name}/unbal", f"{pool['by_source'].get('unbal_train', 0):,}")
            put(f"astag/pool/{name}/bal", f"{pool['by_source'].get('bal_train', 0):,}")
    put("astag/fh/clip", fx(m.get(f"melati:/mnt/data/melati/models/audioset/{AS_FRAMES}"), 4))
    put("astag/gap3", fx(100 * (r["averages"][AS_THREE] - m[AS_M + AS_STAGES[2][1]]), 2))
    put("astag/gap", fx(100 * (r["averages"][AS_SIX] - m[AS_M + AS_STAGES[-1][1]]), 2))
    put("astag/lift4", fx(100 * (m[AS_M + AS_STAGES[-1][1]] - m[AS_M + AS_STAGES[2][1]]), 2))
    # better teachers (audioset/evaluate.py --add-to): our own v3 and v4, BEATs, and PretrainedSED's frame taggers read
    # at clip level, each added to the three
    tb, tm = J("audioset/teachers_beats_fp32.json"), J("audioset/teachers_more_fp32.json")
    if tb and tm:
        put("astag/t/v3v4", fx(tb["averages"][AS_FIVE], 4))
        put("astag/t/beats", fx(tb["averages"][f"beats + {AS_THREE}"], 4))
        psed = [v for k, v in tm["models"].items() if k.startswith("psed:")]
        put("astag/t/psed/lo", fx(min(psed), 3)); put("astag/t/psed/hi", fx(max(psed), 3))
        av = tm["averages"]
        pairs = [(v, av[" + ".join(sorted(p for p in k.split(" + ") if not p.startswith("psed:")))])
                 for k, v in av.items() if "psed:" in k]
        assert all(w < wo for w, wo in pairs), "the text says they lower every average they join"
        put("astag/t/psed/n", str(len(pairs)))
        put("astag/t/psed/three", fx(max(v for k, v in av.items() if "psed:" in k and k.count(" + ") == 3
                                         and AS_M not in k), 4))
    e = J("audioset/ensemble_fp32.json")          # the six's average as released (audioset/release_ensemble.py)
    if e:
        adds = {k: e["map"] - v for k, v in e["without"].items()}
        put("astag/ens", fx(e["map"], 4)); put("astag/ens/bf16", fx(e["map_bf16"], 4))
        put("astag/ens/params", f"{e['params'] / 1e9:.1f}")
        put("astag/ens/ced1", fx(e["members"]["ced-base"]["map"], 4))
        put("astag/ens/add/beats", f"{adds['beats']:.4f}"); put("astag/ens/add/ced", f"{adds['ced-base']:.4f}")
        put("astag/ens/add/eat", f"{adds['eat-large']:.4f}".replace("-", "$-$"))
        put("astag/ens/five", fx(e["without"]["eat-large"], 4))
    g = J("audioset/engine.json")                 # melati_engine.py from the Hub (audioset/check_engine.py)
    if g:
        ev, one = g["eval"], g["one_gpu_graphs"]
        put("astag/eng/map", fx(ev["map"], 4)); put("astag/eng/n", str(len(ev["gpus"])))
        put("astag/eng/multi", f"{ev['clips_per_s']:.0f}"); put("astag/eng/rt", f"{ev['realtime']:,}")
        put("astag/eng/one", f"{one['clips_per_s']:.0f}"); put("astag/eng/lat", f"{one['latency_ms_p50']:.0f}")
        put("astag/eng/ens64", f"{g['ensemble_bf16_b64_clips_per_s']:.0f}")
        put("astag/eng/tflop", f"{g['tflop_per_clip']:.2f}")
    # the Dasheng-1.2B student, stopped halfway: its log, and its checkpoint (the moving average at the last
    # evaluation) in fp32
    log = RES / "audioset" / "train_dasheng_big.jsonl"
    dz = J("audioset/dasheng_half_fp32.json")
    if log.exists() and dz:
        rec = [json.loads(line) for line in log.read_text().splitlines()]
        a = next(x["args"] for x in rec if "args" in x)
        ev = [x for x in rec if x.get("step", 0) > 0]
        put("astag/dash", fx(next(iter(dz["models"].values())), 4))
        put("astag/dash/step", f"{ev[-1]['step']:,}"); put("astag/dash/epoch", f"{ev[-1]['epoch']:g}")
        put("astag/dash/epochs", f"{float(a['epochs']):g}"); put("astag/dash/every", f"{int(a['eval_every']):,}")
        put("astag/dash/last", fx(ev[-1]["map_ema"] - ev[-2]["map_ema"], 4))
    te, we = J("audioset/timing_eval.json"), J("audioset/window_eval.json")
    if te and we:                                  # what places a sound in time (audioset/timing_eval.py)
        put("astag/win/v4", fx(te["frame_map"]["v4, 2 s windows"], 4))
        put("astag/win/clip", fx(we["windows"]["clip"]["shared"], 4))
        put("astag/win/mean", fx(te["frame_map"]["mean of the two frame models"], 4))
        put("astag/win/gated", fx(te["frame_map"]["pretrainedsed x v4 clip"], 4))
    fe = J("audioset/frames_eval_frames1.json")
    if fe:
        k = next(k for k in fe if k.startswith("/"))
        put("astag/frm/ours", fx(fe[k]["shared"], 4)); put("astag/frm/psed", fx(fe["pretrainedsed"]["shared"], 4))
        put("astag/frm/classes", str(fe["classes_shared"])); put("astag/frm/clips", f"{fe['clips']:,}")


if __name__ == "__main__":
    audit(); separation(); vox(); asr(); emotion(); emotion_heads(); emotion_mer2023(); nonverbal(); fillers()
    audioset()
    as_tagger()
    dataset()
    write_numbers()
    abstract_txt()
