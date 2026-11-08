"""Non-verbal sounds in the gaps between aligned words: where by acoustics, what by a classifier.

One definition, shared by the pipeline (stage G, melati/pipeline.py) and its evaluation
(scripts/eval_nonverbal_gaps.py), so the method that was measured is the method that runs.

  where  every gap between two aligned words (and before the first, after the last) that holds sound
         above both the segment's noise floor + FLOOR_DB and its speech level + REL_DB is a candidate,
         with its own times. The level is low enough to keep a breath, far quieter than speech.
  what   the candidate, cut out and padded with silence, goes to the `omni` service (Qwen3-Omni) with a
         closed list of labels as letters A-P: a letter is one token, so the first answer token's
         probabilities are a distribution over the list. Speech, a filler, noise and silence are labels
         too, so "nothing to tag" is an answer.

         Or Melati's own non-verbal heads (`nv_<encoder>` on the tag server, nonverbal/train.py): a frozen
         encoder and a small head trained on PodcastFillers' and VocalSound's timed events, cut and padded
         exactly as a candidate is. Their labels (HEAD_LABELS) are named as LABELS are, so a head's
         distribution is read exactly as Qwen3-Omni's.

A class is tagged where its probability clears a threshold chosen on NonverbalTTS `train` for 0.8 tag
precision (export_config `nv_thresholds`); a class with no threshold is never tagged.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 0.01
# FLOOR_DB was 6: on NonverbalTTS the breaths it missed sat in real gaps, a median 1.7 dB under that threshold
# (nonverbal/gap_rule.py). At 2 dB, 91% of test breaths have a candidate instead of 54%, and the head, retrained on
# what the looser rule finds, tags twice as many breaths at the same precision.
REL_DB, FLOOR_DB, MIN_ACTIVE, PAD_S = -40.0, 2.0, 0.08, 0.3
LABELS = ["speech", "filler", "breath", "laugh", "cough", "sigh", "sniff", "throat clearing", "gasp", "cry", "groan",
          "sneeze", "yawn", "click", "noise", "silence"]
TO_CLASS = {"breath": "breath", "laugh": "laugh", "cough": "cough", "sigh": "sigh", "sniff": "sniff",
            "throat clearing": "throat_clear", "gasp": "gasp", "cry": "cry", "groan": "groan", "sneeze": "sneeze",
            "yawn": "yawn"}
GLOSS = {"speech": "a word or part of a word", "filler": "um, uh, erm or hmm", "breath": "an inhalation or exhalation",
         "click": "a lip smack or tongue click", "noise": "background sound not made by the speaker"}
# what the training data of Melati's own heads labels (nonverbal/build_manifest.py, nonverbal/candidates.py), in the
# heads' output order. `other` is not in LABELS: it is a gap sound no NonverbalTTS annotator tagged, so never a tag.
HEAD_LABELS = ("speech", "filler", "breath", "laugh", "cough", "sigh", "sniff", "throat clearing", "sneeze", "noise",
               "other")
LETTERS = "ABCDEFGHIJKLMNOP"
PROMPT = ("This is a short sound cut out of a recording of someone talking, from a pause between two words, padded "
          "with silence. What is it? Answer with the letter only.\n"
          + "\n".join(f"{LETTERS[i]} {lab}" + (f" ({GLOSS[lab]})" if lab in GLOSS else "") for i, lab in enumerate(LABELS)))


def candidates(x, words, rel_db=REL_DB, floor_db=FLOOR_DB, min_active=MIN_ACTIVE, dilate=0.03, max_len=2.5):
    """Sound in each word gap of `x` (16 kHz): frames above the noise floor + floor_db AND above the speech
    level + rel_db. `words` are [{"s", "e"}] in the same time base. -> [{slot, s, e, level_db}]; slot k =
    before word k (len(words) = after the last)."""
    n = int(FRAME * SR)
    m = len(x) // n
    if m < 10 or not words:
        return []
    db = 10 * np.log10(np.mean(x[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    inside = np.zeros(m, bool)
    for w in words:
        inside[max(0, int(w["s"] / FRAME)):min(m, int(w["e"] / FRAME) + 1)] = True
    speech = float(np.median(db[inside])) if inside.any() else float(np.max(db))
    floor = float(np.percentile(db, 10))
    thr = max(floor + floor_db, speech + rel_db)
    dur = len(x) / SR
    bounds = [(0.0, words[0]["s"])] + [(words[i - 1]["e"], words[i]["s"]) for i in range(1, len(words))] \
        + [(words[-1]["e"], dur)]
    out = []
    for k, (a, b) in enumerate(bounds):
        a2, b2 = a + (dilate if k > 0 else 0.0), b - (dilate if k < len(bounds) - 1 else 0.0)
        fa, fb = int(a2 / FRAME), int(b2 / FRAME)
        if fb - fa < int(min_active / FRAME):
            continue
        act = db[fa:fb] >= thr
        if act.sum() * FRAME < min_active:
            continue
        idx = np.flatnonzero(act)
        s, e = (fa + idx[0]) * FRAME, min((fa + idx[-1] + 1) * FRAME, (fa + idx[0]) * FRAME + max_len)
        out.append({"slot": k, "s": round(s, 3), "e": round(e, 3),
                    "level_db": round(float(np.max(db[fa:fb]) - speech), 1)})
    return out


def clip_of(x, cand, pad_s=PAD_S):
    """The candidate cut out of `x` and padded with silence on both sides."""
    pad = np.zeros(int(pad_s * SR), np.float32)
    return np.concatenate([pad, x[int(cand["s"] * SR):int(cand["e"] * SR)].astype(np.float32), pad])


def dist_of(answer, top):
    """{label: probability} from the first answer token's log-probabilities over the letters; the parsed
    answer when the server returned none."""
    p = {}
    for tok, lp in (top or {}).items():
        t = tok.strip().upper()
        if len(t) == 1 and t in LETTERS[:len(LABELS)]:
            lab = LABELS[LETTERS.index(t)]
            p[lab] = p.get(lab, 0.0) + float(np.exp(lp))
    if not p:
        a = (answer or "").strip().upper()[:1]
        if a and a in LETTERS[:len(LABELS)]:
            p[LABELS[LETTERS.index(a)]] = 1.0
    z = sum(p.values())
    return {k: round(v / z, 4) for k, v in p.items()} if z else {}


def decide(dist, thresholds):
    """The non-verbal class to tag, or None: the most probable non-verbal class, if it clears its threshold
    (it need not beat speech or filler: the threshold is what holds precision). thresholds None = argmax."""
    if not dist:
        return None
    if thresholds is None:
        top = max(dist, key=dist.get)
        return TO_CLASS.get(top)
    nv = {TO_CLASS[k]: v for k, v in dist.items() if k in TO_CLASS}
    if not nv:
        return None
    c = max(nv, key=nv.get)
    t = thresholds.get(c)
    return c if t is not None and nv[c] >= t else None


async def label(client, x, cands, pad_s=PAD_S):
    """Ask the `omni` service about each candidate; -> the candidates with their `dist`."""
    import asyncio
    res = await asyncio.gather(*[client.chat_audio("omni", clip_of(x, c, pad_s), PROMPT, 2, top_logprobs=20)
                                 for c in cands])
    return [dict(c, dist=dist_of(ans, top), by="omni") for c, (ans, top) in zip(cands, res)]


async def label_heads(client, x, cands, models, pad_s=PAD_S, per_request=32):
    """Ask Melati's non-verbal heads (`nv_<encoder>` models on the tag server) about each candidate; the
    distribution is their average. -> the candidates with their `dist`."""
    import asyncio
    clips = [clip_of(x, c, pad_s) for c in cands]
    chunks = [clips[i:i + per_request] for i in range(0, len(clips), per_request)]
    per_model = []
    for m in models:
        parts = await asyncio.gather(*[client.emotion(m, ch) for ch in chunks])
        per_model.append(np.concatenate(parts))
    p = np.mean(per_model, 0)
    return [dict(c, dist={lab: round(float(v), 4) for lab, v in zip(HEAD_LABELS, row)}, by="heads")
            for c, row in zip(cands, p)]


def thresholds_for(cand, nv_thresholds):
    """The thresholds of the labeller that answered: export_config `nv_thresholds` is {labeller: {class: p}}
    ("heads", "omni"); a flat {class: p}, the older form, is Qwen3-Omni's."""
    if any(isinstance(v, dict) for v in nv_thresholds.values()):
        return nv_thresholds.get(cand.get("by", "omni")) or {}
    return nv_thresholds
