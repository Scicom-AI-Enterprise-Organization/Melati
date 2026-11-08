"""Speech emotion classifiers, behind one interface: clips in, probabilities over EMOTIONS out.

Two candidates, compared on labelled sets before either is trusted (scripts/eval_emotion.py):

  meralion   MERaLiON/MERaLiON-SER-v1: a Whisper-medium encoder with LoRA and an ECAPA-TDNN head,
             trained for English (global and Singapore), Chinese, Malay and Tamil. Seven classes
             plus valence/arousal/dominance. The only one of the two trained on Malay.
  e2v        emotion2vec+ large (emotion2vec/emotion2vec_plus_large, FunASR): nine classes from a
             data2vec-style model pretrained on 40k h; English and Chinese pseudo-labelled data.

Both are mapped onto Melati's labels: neutral + the six of `melati.tags.EMOTIONS`. A clip is a
phrase (words between pauses, from the aligner) or a whole segment.
"""
from __future__ import annotations

import numpy as np

LABELS = ("neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted")
SR = 16000


class Meralion:
    name = "meralion"

    def __init__(self, device="cuda", repo="MERaLiON/MERaLiON-SER-v1"):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()
        self.device = device
        id2label = getattr(self.model.config, "id2label", None) or {}
        # the card's usage code and its prose list the classes in different orders: trust the config
        self.order = [str(id2label.get(i, id2label.get(str(i), ""))).lower() for i in range(len(id2label))]
        if not self.order or not all(self.order):
            raise RuntimeError(f"MERaLiON-SER config has no id2label: {id2label}")

    def __call__(self, clips):
        import torch
        # a Whisper encoder reads exactly 30 s of mel (3,000 frames): pad to that, not to the longest clip
        inputs = self.proc([np.asarray(c, dtype=np.float32)[: 30 * SR] for c in clips], sampling_rate=SR,
                           return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        p = logits.float().softmax(-1).cpu().numpy()
        res = np.zeros((len(clips), len(LABELS)), np.float32)
        for j, lab in enumerate(self.order):
            for k, ours in enumerate(LABELS):
                if lab.startswith(ours[:4]):
                    res[:, k] += p[:, j]
        vad = None
        for key in ("dims", "dimensional", "vad", "regression"):
            if isinstance(out, dict) and key in out:
                vad = out[key].float().cpu().numpy()
        return res, vad


class Emotion2Vec:
    name = "e2v"
    _MAP = {"angry": "angry", "disgusted": "disgusted", "fearful": "fearful", "happy": "happy",
            "neutral": "neutral", "sad": "sad", "surprised": "surprised"}

    def __init__(self, device="cuda", repo="emotion2vec/emotion2vec_plus_large"):
        from funasr import AutoModel
        self.model = AutoModel(model=repo, hub="hf" if "/" in repo and not repo.startswith("iic") else "ms",
                               device=device, disable_update=True)

    def __call__(self, clips):
        res = np.zeros((len(clips), len(LABELS)), np.float32)
        out = self.model.generate([np.asarray(c, dtype=np.float32) for c in clips], granularity="utterance",
                                  extract_embedding=False, disable_pbar=True)
        for i, o in enumerate(out):
            for lab, s in zip(o["labels"], o["scores"]):
                eng = lab.split("/")[-1].strip().lower()
                if eng in self._MAP:
                    res[i, LABELS.index(self._MAP[eng])] += float(s)
            tot = res[i].sum()
            if tot > 0:
                res[i] /= tot          # "other" and "<unk>" carry mass that belongs to no label
        return res, None


def load(name: str, device="cuda"):
    """meralion, e2v, h_<encoder> (one of Melati's own emotion heads, melati/ser.py, e.g. h_whisper),
    nv_<encoder> (a non-verbal head: its probabilities are over melati.nv_gaps.HEAD_LABELS, not emotions), or
    nvf_<encoder> (the frame-level non-verbal model, melati/nv_frames.py: per clip, a probability per class per frame)."""
    if name.startswith("nvf_"):
        from melati.nv_frames import FrameNV
        return FrameNV(name[4:], device=device)
    if name.startswith("h_"):
        from melati.ser import HeadSER
        return HeadSER(name[2:], device=device)
    if name.startswith("nv_"):                       # a non-verbal head, served beside the emotion ones
        from melati.ser import HeadNV
        return HeadNV(name[3:], device=device)
    return {"meralion": Meralion, "e2v": Emotion2Vec}[name](device=device)


def phrases(words, max_pause=0.25, min_len=1.0, pad=0.05):
    """Runs of words separated by pauses longer than `max_pause`, each at least `min_len` long
    (a short run is merged into its neighbour). -> [(first word, last word, start, end)]."""
    if not words:
        return []
    runs, cur = [], [0]
    for i in range(1, len(words)):
        if words[i]["s"] - words[i - 1]["e"] > max_pause:
            runs.append(cur); cur = [i]
        else:
            cur.append(i)
    runs.append(cur)
    merged = []
    for r in runs:
        dur = words[r[-1]]["e"] - words[r[0]]["s"]
        if merged and (dur < min_len or words[merged[-1][-1]]["e"] - words[merged[-1][0]]["s"] < min_len):
            merged[-1] = merged[-1] + r
        else:
            merged.append(r)
    return [(r[0], r[-1], max(0.0, words[r[0]]["s"] - pad), words[r[-1]]["e"] + pad) for r in merged]


def spans_from_phrases(phr, probs, thresholds: dict | None = None, default=None):
    """One label per phrase (neutral unless a non-neutral class clears its threshold), then
    consecutive phrases with the same label joined into one span. -> [(label, first, last, conf)].
    A label with no threshold (and no default) is never tagged: its precision was never measured."""
    thresholds = thresholds or {}
    labs = []
    for (a, b, _, _), p in zip(phr, probs):
        k = int(np.argmax(p))
        lab = LABELS[k]
        t = thresholds.get(lab, default)
        if lab != "neutral" and (t is None or p[k] < t):
            lab = "neutral"
        labs.append((lab, a, b, float(p[k])))
    spans = []
    for lab, a, b, c in labs:
        if spans and spans[-1][0] == lab and spans[-1][2] == a - 1:
            spans[-1] = (lab, spans[-1][1], b, min(spans[-1][3], c))
        else:
            spans.append((lab, a, b, c))
    return [s for s in spans if s[0] != "neutral"]
