"""Melati's own heads: a frozen speech encoder, every layer mean-pooled over the clip, and a small trained head.

Two kinds share the machinery: emotion heads (seven emotions, emotion/probe.py) and non-verbal heads (what a sound
in a word gap is, nonverbal/train.py; the labels are melati.nv_gaps.HEAD_LABELS).

One definition, shared by the features the heads were trained on (emotion/extract.py), the heads themselves
(emotion/probe.py) and the tag server that runs them (serve/tag_server.py through melati.emotion.load), so what
was measured is what runs. Pooling covers the clip's own frames only: Whisper pads every input to 30 s.

Encoders (EmoBox found the Whisper-large-v3 encoder the strongest SER backbone, WavLM-large the next):
  whisper     openai/whisper-large-v3
  whisper_ft  the released Melati ASR, Scicom-intl/Melati-Whisper-Large-V3 at a pinned revision
              (MELATI_SER_WHISPER_FT). The first heads read an earlier candidate's encoder, which was deleted
              with the box on 2026-10-29; the heads trained since read the ASR the pipeline transcribes with.
  wavlm       microsoft/wavlm-large
  w2vbert     facebook/w2v-bert-2.0
  e2v         emotion2vec+ large, its utterance embedding (one layer)
  ast         MIT/ast-finetuned-audioset-10-10-0.4593: AudioSet's 527 sound classes include laughter, breathing,
              cough, sigh, sniff, sneeze and throat clearing, the non-verbal heads' business
"""
from __future__ import annotations

import os

import numpy as np

SR = 16000
MAX_S = 30.0
LABELS = ("neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted")
DATA = os.environ.get("MELATI_DATA", "/root/melati/data")        # data, models, results: deploy/env.sh
WHISPER_FT = os.environ.get("MELATI_SER_WHISPER_FT", "Scicom-intl/Melati-Whisper-Large-V3")
WHISPER_FT_REVISION = os.environ.get("MELATI_SER_WHISPER_FT_REVISION", "cdbb981c6ce8d87b338ed9503ef9a51f24a2a6aa")
HEADS_DIR = os.environ.get("MELATI_SER_HEADS", f"{DATA}/models/ser_heads")
NV_HEADS_DIR = os.environ.get("MELATI_NV_HEADS", f"{DATA}/models/nv_heads")


def whisper_encoder(name, device="cuda", revision=None):
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
    revision = None if os.path.isdir(name) else revision                # a local folder has no revisions
    fe = WhisperFeatureExtractor.from_pretrained(name, revision=revision)
    enc = WhisperModel.from_pretrained(name, revision=revision, dtype=torch.float16).encoder.to(device).eval()

    def run(batch):
        f = fe(batch, sampling_rate=SR, return_tensors="pt").input_features.to(device).half()
        hs = enc(f, output_hidden_states=True).hidden_states            # (layers + 1) x (B, 1500, D)
        h = torch.stack(hs, 1)                                            # B, L, T, D
        n = [max(1, min(1500, int(len(x) / SR * 50))) for x in batch]    # 50 encoder frames per second
        return np.stack([h[i, :, :k].float().mean(1).cpu().numpy() for i, k in enumerate(n)])
    return run


def hf_ssl(name, device="cuda", batched=True, max_batch=16, max_ratio=1.3):
    """Every layer of a wav2vec-style encoder (WavLM, w2v-BERT), each clip mean-pooled over its own frames.

    The heads were trained on features computed one clip at a time (batched=False). Batched, clips of similar length
    share one forward, padded and masked, with the same arithmetic for every frame of a clip: w2v-BERT's depthwise
    convolutions are causal and see padding masked to zero, WavLM zeroes padded frames before its positional
    convolution, attention never reads a padded key, and each clip is pooled over exactly the frames it has alone
    (scripts/check_ssl_batching.py measures the difference: fp16 rounding). One clip at a time left a shared GPU
    mostly waiting on hundreds of small kernels; w2v-BERT was the slowest model in the tag server by four times."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    to_dev = lambda inp: {k: (v.to(device).half() if v.is_floating_point() else v.to(device))  # noqa: E731
                          for k, v in inp.items()}

    def frames_alone(n):
        """The frames a clip of n samples has when it is encoded by itself."""
        if "input_features" in fe.model_input_names:                      # w2v-BERT: 25 ms fbank every 10 ms,
            k = 1 + (n - 400) // 160                                        # padded to even, stacked in pairs
            return (k + 1) // 2
        return int(model._get_feat_extract_output_lengths(torch.tensor(n)))

    def one_at_a_time(batch):
        out = []
        for x in batch:
            hs = model(**to_dev(fe(x, sampling_rate=SR, return_tensors="pt")), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy())
        return np.stack(out)

    def together(xs):
        hs = model(**to_dev(fe(xs, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)),
                   output_hidden_states=True).hidden_states
        w = torch.zeros(hs[0].shape[:2], device=hs[0].device)            # B, T: 1/frames over each clip's own
        for i, x in enumerate(xs):
            k = min(frames_alone(len(x)), w.shape[1])
            w[i, :k] = 1.0 / k
        return list(torch.stack([torch.einsum("btd,bt->bd", h.float(), w) for h in hs], 1).cpu().numpy())

    def run(batch):
        if not batched:
            return one_at_a_time(batch)
        order = sorted(range(len(batch)), key=lambda i: len(batch[i]))
        out, group = [None] * len(batch), []
        for i in order + [None]:
            if group and (i is None or len(group) == max_batch or len(batch[i]) > max_ratio * len(batch[group[0]])):
                for j, f in zip(group, together([batch[j] for j in group])):
                    out[j] = f
                group = []
            if i is not None:
                group.append(i)
        return np.stack(out)
    return run


def ast_encoder(name="MIT/ast-finetuned-audioset-10-10-0.4593", device="cuda"):
    """Every layer of AST, mean-pooled over the patches that cover the clip. AST reads a fixed 10.24 s of filter
    bank, so a short clip is mostly padding; its patches are flattened frequency-major (f * t_dim + t, after
    the CLS and distillation tokens) and patch t covers frames [10 t, 10 t + 16)."""
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    fe = ASTFeatureExtractor.from_pretrained(name)
    model = ASTModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    c = model.config
    f_dim = (c.num_mel_bins - c.patch_size) // c.frequency_stride + 1
    t_dim = (c.max_length - c.patch_size) // c.time_stride + 1

    def run(batch):
        x = fe(batch, sampling_rate=SR, return_tensors="pt").input_values.to(device).half()
        h = torch.stack(model(x, output_hidden_states=True).hidden_states, 1)          # B, L, 2 + F*T, D
        h = h[:, :, 2:].reshape(h.shape[0], h.shape[1], f_dim, t_dim, h.shape[-1])
        out = []
        for i, clip in enumerate(batch):
            frames = 1 + max(0, len(clip) - 400) // 160                                 # 25 ms windows, 10 ms hop
            nt = max(1, min(t_dim, (frames - c.patch_size) // c.time_stride + 1))
            out.append(h[i, :, :, :nt].float().mean((1, 2)).cpu().numpy())
        return np.stack(out)
    return run


def e2v_embedding(device="cuda"):
    from funasr import AutoModel                      # as melati/emotion.py loads it for its own predictions
    m = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", device=device, disable_update=True)

    def run(batch):
        res = m.generate([np.asarray(x, np.float32) for x in batch], granularity="utterance",
                         extract_embedding=True, disable_pbar=True)
        return np.stack([np.asarray(r["feats"], np.float32).reshape(1, -1) for r in res])
    return run


ENCODERS = {"whisper": lambda d: whisper_encoder("openai/whisper-large-v3", d),
            "whisper_ft": lambda d: whisper_encoder(WHISPER_FT, d, WHISPER_FT_REVISION),
            "wavlm": lambda d: hf_ssl("microsoft/wavlm-large", d),
            "w2vbert": lambda d: hf_ssl("facebook/w2v-bert-2.0", d),
            "e2v": e2v_embedding,
            "ast": lambda d: ast_encoder(device=d)}


def prepare(clips):
    """Clips as the features were computed: at most 30 s, at least 0.1 s."""
    out = []
    for x in clips:
        x = np.asarray(x, np.float32)[: int(MAX_S * SR)]
        out.append(x if len(x) >= SR // 10 else np.pad(x, (0, SR // 10 - len(x))))
    return out


def head_module(layers, dim, hidden=256, n=len(LABELS), drop=0.2):
    import torch.nn as nn

    class Head(nn.Module):
        """A softmax mix of the encoder's layers, LayerNorm, a hidden layer, seven outputs."""

        def __init__(self):
            super().__init__()
            import torch
            self.mix = nn.Parameter(torch.zeros(layers))
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(hidden, n))

        def forward(self, x):                       # x: B, L, D
            return self.net((self.mix.softmax(0)[None, :, None] * x).sum(1))
    return Head()


class Head:
    """One encoder and its trained head, behind the call melati.emotion's models share: clips -> (probs, None).
    The checkpoint names its labels, and they must be the ones this kind of head is read with."""

    def __init__(self, encoder: str, labels, heads_dir: str, prefix: str, device="cuda"):
        import torch
        ck = torch.load(os.path.join(heads_dir, f"{encoder}.pt"), map_location="cpu")
        assert tuple(ck["labels"]) == tuple(labels), (ck["labels"], labels)
        self.name, self.labels = f"{prefix}_{encoder}", tuple(labels)
        self.head = head_module(ck["layers"], ck["dim"], n=len(labels)).to(device).eval()
        self.head.load_state_dict(ck["state"])
        self.run = ENCODERS[encoder](device)
        self.device = device

    def __call__(self, clips):
        import torch
        with torch.inference_mode():
            f = torch.from_numpy(self.run(prepare(clips)).astype(np.float16)).to(self.device).float()
            return self.head(f).softmax(-1).cpu().numpy().astype(np.float32), None


def HeadSER(encoder: str, device="cuda", heads_dir=HEADS_DIR):
    """An emotion head (h_<encoder>)."""
    return Head(encoder, LABELS, heads_dir, "h", device)


def HeadNV(encoder: str, device="cuda", heads_dir=NV_HEADS_DIR):
    """A non-verbal head (nv_<encoder>): what a sound cut out of a word gap is."""
    from melati.nv_gaps import HEAD_LABELS
    return Head(encoder, HEAD_LABELS, heads_dir, "nv", device)
