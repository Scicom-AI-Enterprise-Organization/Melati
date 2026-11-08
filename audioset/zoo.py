#!/usr/bin/env python3
"""AudioSet taggers released by others, each behind one call: 16 kHz clips (10 s rows) -> (N, 527) probabilities in
AudioSet's class order (audioset/data.py classes). Scored on the same clips by audioset/evaluate.py and, averaged,
the teachers a Melati model is distilled from. RUNS ON THE BOX, in .venv-sed.

  ced-base    mispeech/ced-base (Dinkel et al., ICASSP 2024; Apache-2.0): 64 mel bins, its own feature extractor
  eat-large   worstchan/EAT-large_epoch20_finetune_AS2M (Chen et al., IJCAI 2024; MIT)
  eat-base    worstchan/EAT-base_epoch30_finetune_AS2M (MIT)
  sslam       ta012/SSLAM_AS2M_Finetuned (Alex et al., ICLR 2025; MIT): EAT's architecture, pre-trained on mixtures
  ast         MIT/ast-finetuned-audioset-10-10-0.4593 (Gong et al., Interspeech 2021; BSD-3-Clause)

  beats       BEATs iter3+ fine-tuned on AS2M, cpt2 (Chen et al., ICML 2023; MIT): microsoft/unilm's code in
              $MELATI_ROOT/ext/beats, the checkpoint from two Hugging Face mirrors that agree byte for byte (Microsoft's
              OneDrive link refuses scripts; deploy/box_beats_fetch.sh), loaded weights_only

PretrainedSED's frame taggers read at clip level, each class's highest probability over the clip's 40 ms frames, for
the 447 classes they have (NaN in the other 80; audioset/evaluate.py averages a model only where it has the class):

  psed:BEATs, psed:ATST-F, psed:M2D   fschmid56/PretrainedSED (Schmid et al., ICASSP 2025; MIT), fine-tuned on
                                      AudioSet's strongly labelled training clips (nonverbal/audioset_eval.py loads them)

and one encoder with no AudioSet head of its own, only ever a student's starting point (audioset/train.py):

  dasheng-1.2b  mispeech/dasheng-1.2B (Dinkel et al., Interspeech 2024; Apache-2.0): 1.2B parameters, self-supervised
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SR, classes  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh
MELATI_ROOT = os.environ.get("MELATI_ROOT", "/root/melati")        # environments, checkouts: deploy/env.sh

BF16 = True        # bf16 autocast: several times the fp32 speed; audioset/evaluate.py --fp32 checks it costs no mAP
REPOS = {"ced-base": "mispeech/ced-base", "eat-large": "worstchan/EAT-large_epoch20_finetune_AS2M",
         "dasheng-1.2b": "mispeech/dasheng-1.2B", "eat-base": "worstchan/EAT-base_epoch30_finetune_AS2M", "sslam": "ta012/SSLAM_AS2M_Finetuned",
         "ast": "MIT/ast-finetuned-audioset-10-10-0.4593"}


def _order(id2label):
    """A model's output index -> AudioSet's index, by display name (None when the model keeps AudioSet's order)."""
    names = [str(v) for v in (id2label or {}).values()]
    if not names or all(n.startswith("LABEL_") for n in names):       # no names: the model keeps AudioSet's order
        return None
    by_name = {n: i for i, _, n in classes()}
    order = [by_name.get(id2label[k] if k in id2label else id2label[str(k)]) for k in range(len(id2label))]
    if any(o is None for o in order):
        missing = [id2label.get(k, id2label.get(str(k))) for k, o in enumerate(order) if o is None]
        raise SystemExit(f"labels not in AudioSet's list: {missing[:5]}")
    return None if order == list(range(len(order))) else np.argsort(order)


_BANKS = {}


def kaldi_fbank(w):
    """torchaudio.compliance.kaldi.fbank as EAT and SSLAM call it (128 bins, 25 ms Hanning windows every 10 ms, DC
    removed per frame, pre-emphasis 0.97, no dither, log mel power floored at float eps), for a batch at once on the GPU:
    (B, samples) at 16 kHz, each row already centred -> (B, frames, 128). The library does one waveform a call, which
    held EAT-large to 70 clips a second."""
    from torchaudio.compliance.kaldi import get_mel_banks
    key = (w.device, w.dtype)
    if key not in _BANKS:
        banks, _ = get_mel_banks(128, 512, float(SR), 20.0, 0.0, 100.0, -500.0, 1.0)
        _BANKS[key] = (torch.nn.functional.pad(banks, (0, 1)).to(w.device, torch.float32),     # (128, 257)
                       torch.hann_window(400, periodic=False, device=w.device, dtype=torch.float32))
    banks, win = _BANKS[key]
    fr = w.float().unfold(1, 400, 160)                                         # snip_edges: whole frames only
    fr = fr - fr.mean(-1, keepdim=True)
    fr = fr - 0.97 * torch.cat([fr[..., :1], fr[..., :-1]], -1)
    spec = torch.fft.rfft(fr * win, n=512).abs().pow(2)                         # (B, frames, 257)
    return torch.clamp(spec @ banks.T, min=torch.finfo(torch.float32).eps).log()


class EATLike:
    """EAT and SSLAM: Kaldi fbank, 128 bins, 1024 frames (10.24 s), normalised as their cards say; model(x) -> logits."""

    def __init__(self, repo, revision=None):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(repo, trust_remote_code=True, revision=revision).eval().cuda()
        self.order = _order(getattr(self.m.config, "id2label", None) if len(getattr(self.m.config, "id2label", {}) or {})
                            == 527 else None)

    @torch.no_grad()
    def __call__(self, xs):
        w = torch.as_tensor(np.stack([np.asarray(x, np.float32) for x in xs])).cuda()
        f = kaldi_fbank(w - w.mean(1, keepdim=True))
        f = torch.nn.functional.pad(f, (0, 0, 0, max(0, 1024 - f.shape[1])))[:, :1024]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            out = self.m(((f + 4.268) / (4.569 * 2))[:, None])
        logits = out if torch.is_tensor(out) else (out.logits if hasattr(out, "logits") else out[0])
        p = torch.sigmoid(logits.float()).cpu().numpy()
        return p[:, self.order] if self.order is not None else p


class HFClassifier:
    """A transformers audio classifier with its own feature extractor (CED, AST)."""

    def __init__(self, repo):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.fe = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True)
        self.m = AutoModelForAudioClassification.from_pretrained(repo, trust_remote_code=True).eval().cuda()
        self.order = _order(self.m.config.id2label)

    @torch.no_grad()
    def __call__(self, xs):
        inp = self.fe([np.asarray(x, np.float32) for x in xs], sampling_rate=SR, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            out = self.m(**{k: v.cuda() for k, v in inp.items()}).logits.float()
        p = (out if (out.min() >= 0 and out.max() <= 1) else torch.sigmoid(out)).cpu().numpy()  # CED may emit probs
        return p[:, self.order] if self.order is not None else p


class Distilled(EATLike):
    """A student audioset/train.py saved (best.pt: the averaged weights), on the EAT architecture it started from."""

    def __init__(self, ck):
        super().__init__(REPOS[ck["init"]])
        self.m.load_state_dict(ck["state"])
        self.m.eval()


class DistilledDasheng:
    """A Dasheng student audioset/train.py saved: Dasheng's encoder and the head train.DashengTagger puts on it, fed
    Dasheng's own mel spectrogram."""

    def __init__(self, ck):
        from train import student
        self.m, self.feat = student(ck["init"], ck["state"])
        self.m = self.m.eval().cuda()

    @torch.no_grad()
    def __call__(self, xs):
        w = torch.as_tensor(np.stack([np.asarray(x, np.float32) for x in xs])).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            logits = self.m(self.feat(w))
        return torch.sigmoid(logits.float()).cpu().numpy()


BEATS = Path(f"{MELATI_DATA}/models/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")


class BEATsTagger:
    """BEATs fine-tuned on AudioSet: 16 kHz waveforms in (its own filterbanks inside), its sigmoid probabilities out,
    put in AudioSet's order by the checkpoint's label_dict (each output's mid or name)."""

    def __init__(self, path=BEATS):
        sys.path.insert(0, f"{MELATI_ROOT}/ext/beats")
        from BEATs import BEATs, BEATsConfig
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.m = BEATs(BEATsConfig(ck["cfg"]))
        self.m.load_state_dict(ck["model"])
        self.m = self.m.eval().cuda()
        by = {}
        for i, mid, name in classes():
            by[mid], by[name] = i, i
        ld = ck["label_dict"]
        self.order = np.array([by[ld[k]] for k in range(len(ld))])
        assert sorted(self.order.tolist()) == list(range(527)), "label_dict does not name AudioSet's 527 classes"

    @torch.no_grad()
    def __call__(self, xs):
        w = torch.as_tensor(np.stack([np.asarray(x, np.float32) for x in xs])).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            p = self.m.extract_features(w, padding_mask=torch.zeros_like(w, dtype=torch.bool))[0]
        out = np.empty((len(xs), 527), np.float32)
        out[:, self.order] = p.float().cpu().numpy()
        return out


class PSEDClip:
    """A PretrainedSED frame tagger at clip level: per clip, each of its classes' highest frame probability, in
    AudioSet's order, NaN where it has no such class."""

    def __init__(self, name):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nonverbal"))
        from audioset_eval import Tagger, class_index
        self.tagger = Tagger(name, batch=32)
        ours = {n: i for i, _, n in classes()}
        self.pairs = [(theirs, ours[n]) for n, theirs in class_index().items() if n in ours]

    def __call__(self, xs):
        out = np.full((len(xs), 527), np.nan, np.float32)
        for k, p in enumerate(self.tagger([np.asarray(x, np.float32) for x in xs])):     # (447, frames) each
            peak = p.max(1)
            for theirs, j in self.pairs:
                out[k, j] = peak[theirs]
        return out


def load(name):
    """A released model by its name in REPOS, a PretrainedSED tagger (psed:<name>), or a student by the path of its
    checkpoint (melati:<path>)."""
    if name.startswith("psed:"):
        return PSEDClip(name.split(":", 1)[1])
    if name == "beats":
        return BEATsTagger()
    if name.startswith("melati:"):
        ck = torch.load(name.split(":", 1)[1], map_location="cpu", weights_only=False)
        return (DistilledDasheng if ck["init"].startswith("dasheng") else Distilled)(ck)
    repo = REPOS[name]
    return HFClassifier(repo) if name in ("ced-base", "ast") else EATLike(repo)


def main():
    """Smoke test: load each model, tag a second of silence and a tone, print the output's shape and range."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    names = [n for _, _, n in classes()]
    t = np.arange(SR * 10) / SR
    xs = [np.zeros(SR * 10, np.float32), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)]
    for name in a.models:
        try:
            p = load(name)(xs)
            print(f"{name}: {p.shape}, range {p.min():.3f}..{p.max():.3f}, tone -> {names[int(p[1].argmax())]}", flush=True)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)


if __name__ == "__main__":
    main()
