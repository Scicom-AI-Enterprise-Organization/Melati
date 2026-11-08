import json

import numpy as np

from melati.stage_export import DEFAULTS, build, release_drop
from melati.tags import well_formed


def stems(seconds=20.0, speech_db=-20.0, bg_db=-60.0):
    n = int(seconds / 0.05)
    return {"frame_s": np.float32(0.05), "speech": np.full(n, speech_db, np.float16),
            "music": np.full(n, bg_db, np.float16), "sfx": np.full(n, bg_db, np.float16)}


def test_a_full_segment():
    seg = {"seg": "r1_0001", "rec": "r1", "speaker": "S01", "start": 10.0, "end": 16.0,
           "forced_cut": False, "overlap_s_removed": 0.0, "prev": None, "next": None}
    words = [{"w": "hello", "s": 0.3, "e": 0.7, "c": 0.9}, {"w": "sorry", "s": 1.6, "e": 2.0, "c": 0.9},
             {"w": "I", "s": 2.6, "e": 2.7, "c": 0.9}, {"w": "got", "s": 2.7, "e": 2.9, "c": 0.9},
             {"w": "it", "s": 2.9, "e": 3.1, "c": 0.9}]
    al = {"seg": "r1_0001", "words": words, "score": 0.9, "first_word_s": 0.3, "tail_s": 2.9}
    asr = {"seg": "r1_0001", "text": "hello sorry I got it", "language": "en"}
    mos = {"seg": "r1_0001", "mos_speech": 3.1, "mos_mix": 3.3, "dnsmos_speech": {"OVRL": 2.9},
           "dnsmos_mix": {"OVRL": 3.0}}
    tag = {"seg": "r1_0001",
           # a cough in the first word gap and a laugh after the last word, as Qwen3-Omni labelled them
           "nv_gaps": [{"slot": 1, "s": 0.9, "e": 1.3, "dist": {"cough": 0.97, "speech": 0.03}},
                       {"slot": 5, "s": 3.4, "e": 4.0, "dist": {"laugh": 0.6, "speech": 0.4}}],
           # the two released classifiers, all a cluster without Melati's own emotion heads serves
           "emotion": {m: {"segment": [0.02, 0.9, 0.02, 0.02, 0.02, 0.01, 0.01],
                           "phrases": [[0, 4, 0.25, 3.15, [0.02, 0.9, 0.02, 0.02, 0.02, 0.01, 0.01]]]}
                       for m in ("meralion", "e2v")},
           "rate": []}
    fill = {"seg": "r1_0001", "gaps": [{"s": 2.1, "e": 2.45, "text": "Erm.", "kind": "filler"}]}
    row, (a, b), version = build(seg, asr, al, mos, tag, fill, stems(), dict(DEFAULTS))
    assert row["tagged_text"] == "<happy>hello [coughs] sorry erm I got it</happy> [laughs]", row["tagged_text"]
    assert well_formed(row["tagged_text"])
    assert row["text"] == "hello sorry erm I got it"
    assert row["n_fillers"] == 1 and row["n_events"] == 2
    assert version == "mix"                       # nothing to remove, and MOS prefers the mix
    assert abs(a - 10.15) < 1e-6 and abs(b - 14.15) < 1e-6   # first sound - 0.15 .. last sound + 0.15
    ann = json.loads(row["annotation"])
    assert ann["words"][0]["start"] == 0.15


def test_music_under_speech_keeps_the_separated_vocal_version():
    seg = {"seg": "r1_0002", "rec": "r1", "speaker": "S02", "start": 0.0, "end": 3.0}
    al = {"words": [{"w": "yes", "s": 0.5, "e": 0.9, "c": 0.9}], "score": 0.9, "tail_s": 2.1}
    row, _, version = build(seg, {"text": "yes"}, al, {"mos_vocal": 3.0, "mos_mix": 3.4}, {}, {},
                            stems(bg_db=-30.0), dict(DEFAULTS))
    assert version == "vocal"                     # music only 10 dB under the voice
    assert row["mos_vocal"] == 3.0 and row["audio_version"] == "vocal"


def test_effects_alone_never_trigger_separation():
    """A laugh or a cough lives in the EFFECTS stem: loud effects under quiet music keep the mix."""
    seg = {"seg": "r1_0003", "rec": "r1", "speaker": "S02", "start": 0.0, "end": 3.0}
    al = {"words": [{"w": "yes", "s": 0.5, "e": 0.9, "c": 0.9}], "score": 0.9, "tail_s": 2.1}
    st = stems(bg_db=-60.0)
    st["sfx"] = np.full_like(st["sfx"], -22.0)    # a laugh nearly as loud as the speech
    _, _, version = build(seg, {"text": "yes"}, al, {"mos_vocal": 3.0, "mos_mix": 3.0}, {}, {}, st, dict(DEFAULTS))
    assert version == "mix"


def test_an_uncalibrated_emotion_is_not_tagged():
    from melati.emotion import spans_from_phrases
    disgusted = [0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.95]     # no held-out positives: never calibrated
    assert "disgusted" not in DEFAULTS["emotion_thresholds"]
    assert spans_from_phrases([(0, 2, 0.0, 1.0)], [disgusted], DEFAULTS["emotion_thresholds"],
                              DEFAULTS["emotion_default_threshold"]) == []
    happy = [0.02, 0.9, 0.02, 0.02, 0.02, 0.01, 0.01]
    assert spans_from_phrases([(0, 2, 0.0, 1.0)], [happy], DEFAULTS["emotion_thresholds"],
                              DEFAULTS["emotion_default_threshold"])[0][0] == "happy"


def test_emotion_is_tagged_at_the_thresholds_of_the_models_averaged():
    seg = {"seg": "r1_0003", "rec": "r1", "speaker": "S01", "start": 0.0, "end": 3.0}
    words = [{"w": "that", "s": 0.3, "e": 0.6, "c": 0.9}, {"w": "is", "s": 0.6, "e": 0.8, "c": 0.9},
             {"w": "great", "s": 0.8, "e": 1.3, "c": 0.9}]
    al = {"seg": "r1_0003", "words": words, "score": 0.9, "first_word_s": 0.3, "tail_s": 1.7}
    asr = {"seg": "r1_0003", "text": "that is great", "language": "en"}

    def tagged(models, happy):
        p = [0.9 - happy, happy, 0.02, 0.02, 0.02, 0.02, 0.02]
        tag = {"seg": "r1_0003", "emotion": {m: {"segment": p, "phrases": [[0, 2, 0.25, 1.45, p]]} for m in models}}
        row = build(seg, asr, al, {}, tag, None, stems(3.0), dict(DEFAULTS))[0]
        return row["tagged_text"], row["emotion_segment"]
    assert tagged(DEFAULTS["emotion_models"], 0.7) == ("<happy>that is great</happy>", "happy")
    assert tagged(["meralion", "e2v"], 0.7) == ("that is great", "happy")        # the pair tags happy at 0.88
    assert tagged(["meralion", "e2v"], 0.89)[0] == "<happy>that is great</happy>"
    assert tagged(["meralion"], 0.89)[0] == "that is great"                     # MERaLiON alone never tags happy
    assert tagged(["h_whisper", "meralion"], 0.89) == ("that is great", "happy")   # a set nobody measured


def test_detector_events_are_not_tags_unless_asked():
    seg = {"seg": "r1_0004", "rec": "r1", "speaker": "S02", "start": 0.0, "end": 3.0}
    al = {"words": [{"w": "yes", "s": 0.5, "e": 0.9, "c": 0.9}], "score": 0.9, "tail_s": 2.1}
    tag = {"sed": [{"label": "laugh", "s": 1.2, "e": 1.8, "p": 0.9}],
           "nv_gaps": [{"slot": 1, "s": 1.2, "e": 1.8, "dist": {"breath": 0.99, "speech": 0.01}}]}
    row, _, _ = build(seg, {"text": "yes"}, al, {"mos_vocal": 3.0, "mos_mix": 3.0}, tag, {}, stems(), dict(DEFAULTS))
    assert "[" not in row["tagged_text"], row["tagged_text"]    # PANNs off; breath has no threshold
    cfg = dict(DEFAULTS, sed_tags=True)
    row, _, _ = build(seg, {"text": "yes"}, al, {"mos_vocal": 3.0, "mos_mix": 3.0}, tag, {}, stems(), cfg)
    assert "[laughs]" in row["tagged_text"]


def test_a_head_breath_is_tagged_at_the_heads_threshold():
    seg = {"seg": "r1_0005", "rec": "r1", "speaker": "S02", "start": 0.0, "end": 3.0}
    al = {"words": [{"w": "yes", "s": 0.5, "e": 0.9, "c": 0.9}], "score": 0.9, "tail_s": 2.1}
    thr = DEFAULTS["nv_thresholds"]["heads"]["breath"]

    def tagged(p, by):
        dist = {"breath": p, "speech": 1 - p}
        tag = {"nv_gaps": [{"slot": 1, "s": 1.2, "e": 1.6, "dist": dist, "by": by}]}
        row, _, _ = build(seg, {"text": "yes"}, al, {"mos_vocal": 3.0, "mos_mix": 3.0}, tag, {}, stems(), dict(DEFAULTS))
        return row["tagged_text"]
    assert "[breathes]" in tagged(min(1.0, thr + 0.05), "heads")
    assert "[" not in tagged(thr - 0.05, "heads")
    assert "[" not in tagged(0.99, "omni")                      # Qwen3-Omni never earned a breath threshold


def test_a_frame_laugh_over_speech_is_tagged_once():
    """The frame model's laughs become tags, over words too; one next to a gap laugh is not a second tag."""
    seg = {"seg": "r1_0006", "rec": "r1", "speaker": "S02", "start": 0.0, "end": 3.0}
    words = [{"w": "that", "s": 0.4, "e": 0.8, "c": 0.9}, {"w": "is", "s": 0.8, "e": 1.0, "c": 0.9},
             {"w": "funny", "s": 1.0, "e": 1.6, "c": 0.9}]
    al = {"words": words, "score": 0.9, "tail_s": 1.6}
    thr = DEFAULTS["nv_frame_thresholds"]["laugh"]
    laugh = [0.0] * 30
    for f in range(9, 15):                                         # 0.9 to 1.5 s: over "is funny"
        laugh[f] = min(1.0, thr + 0.05)

    def tagged(gap=None, frames=laugh):
        tag = {"nv_frames": {"model": "nvf_whisper", "fps": 10, "laugh": frames, "cough": [0.0] * 30},
               "nv_gaps": gap or []}
        row, _, _ = build(seg, {"text": "that is funny"}, al, {"mos_vocal": 3.0, "mos_mix": 3.0}, tag, {}, stems(),
                          dict(DEFAULTS))
        return row["tagged_text"]
    assert tagged().count("[laughs]") == 1
    assert "[" not in tagged(frames=[round(thr - 0.05, 3)] * 30)
    beside = [{"slot": 3, "s": 1.6, "e": 2.0, "dist": {"laugh": 0.99, "speech": 0.01}, "by": "omni"}]
    assert tagged(gap=beside).count("[laughs]") == 1               # the gap labeller's laugh, not a second one


def test_frame_events_join_short_gaps():
    from melati.nv_frames import events
    p = np.array([0, .9, .95, 0, .99, 0, 0, .97, .98, 0, 0, 0, .99, 0])
    assert [(a, b) for a, b, _ in events(p, 0.9)] == [(1, 9)]     # two quiet frames joined, a lone frame dropped


def test_squeezed_words_are_counted():
    seg = {"seg": "r1_0004", "rec": "r1", "speaker": "S01", "start": 0.0, "end": 3.0}
    al = {"words": [{"w": "ya", "s": 0.5, "e": 0.9, "c": 0.9}, {"w": "lah", "s": 0.9, "e": 0.92, "c": 0.4},
                    {"w": "kan", "s": 0.92, "e": 0.94, "c": 0.4}], "score": 0.5, "tail_s": 2.0}
    row, _, _ = build(seg, {"text": "ya lah kan"}, al, {}, {}, {}, stems(), dict(DEFAULTS))
    assert row["squeezed_words"] == 2
    al["words"] = [dict(w, s=np.float64(w["s"]), e=np.float64(w["e"])) for w in al["words"]]   # as the aligner's arrays give
    row, _, _ = build(seg, {"text": "ya lah kan"}, al, {}, {}, {}, stems(), dict(DEFAULTS))
    json.dumps(row)                               # export.jsonl is JSON: no numpy integer may reach the row


def test_the_release_keeps_only_languages_the_model_heard():
    cfg = dict(DEFAULTS)
    ms = {"language": "ms", "text": "saya pun tak tahu"}
    assert release_drop(ms, {"language": "ms", "detected_language": "id"}, {"id": 0.6, "ms": 0.4}, cfg) is None
    ta = {"language": "ta", "text": "அது"}
    assert release_drop(ta, {"language": "ta", "detected_language": "ar"}, {"ms": 0.9}, cfg) == "language_fallback"
    assert release_drop(ta, {"language": "ta", "detected_language": "ta"}, {"ms": 0.9, "ta": 0.1}, cfg) == "language_stray"
    assert release_drop(ta, {"language": "ta", "detected_language": "ta"}, {"ta": 0.8, "en": 0.2}, cfg) is None
    zh = {"language": "zh", "text": "ni hao ma"}
    assert release_drop(zh, {"language": "zh", "detected_language": "zh"}, {"zh": 0.9}, cfg) == "wrong_script"
    assert release_drop({"language": "en", "text": "شكرا"}, {"detected_language": "en"}, {"en": 1.0}, cfg) == "wrong_script"
    assert release_drop({"language": "en", "text": "..."}, {"detected_language": "en"}, {"en": 1.0}, cfg) == "no_text"
    assert release_drop({"language": "ms", "text": "17, 18, 19."}, {"detected_language": "ms"}, {"ms": 1.0}, cfg) is None


def test_audioset_sniffs_and_whispering_become_tags():
    seg = {"seg": "r1_0005", "rec": "r1", "speaker": "S01", "start": 0.0, "end": 4.0}
    al = {"words": [{"w": "boleh", "s": 0.2, "e": 0.6, "c": 0.9}, {"w": "tak", "s": 1.6, "e": 1.9, "c": 0.9},
                    {"w": "rahsia", "s": 2.0, "e": 2.5, "c": 0.9}, {"w": "ni", "s": 2.6, "e": 2.8, "c": 0.9}],
          "score": 0.9, "tail_s": 1.2}
    au = {"fps": 10, "sniff": [0.0] * 40, "whisper": [0.0] * 40, "throat_clear": [0.0] * 40}
    au["sniff"][9:13] = [0.4] * 4                 # 0.9 to 1.3 s, in the gap after "boleh"
    au["whisper"][19:28] = [0.8] * 9              # 1.9 to 2.8 s, over "rahsia ni"
    row, _, _ = build(seg, {"text": "boleh tak rahsia ni"}, al, {}, {"audioset": au}, {}, stems(), dict(DEFAULTS))
    assert row["tagged_text"] == "boleh [sniffs] tak <whisper>rahsia ni</whisper>", row["tagged_text"]
    assert well_formed(row["tagged_text"])
    ev = json.loads(row["annotation"])["events"]
    assert [e["kind"] for e in ev] == ["sniffs"] and ev[0]["source"] == "audioset"
