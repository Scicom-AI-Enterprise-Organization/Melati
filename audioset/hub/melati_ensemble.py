"""Melati-AudioSet-Ensemble: six AudioSet taggers run on the same clip, their probabilities averaged.

Every member is built here in plain PyTorch and only its weights are downloaded: CED-base, EAT-large and SSLAM from
their own Hugging Face repos, Melati-AudioSet-EAT-Large v3 and v4 from ours, each at the revision ensemble.json pins
(config.json and model.safetensors, nothing else), and BEATs from this repo (beats.safetensors, with microsoft/unilm's
code for it in melati_beats/, MIT). No repo's own code is run, so it needs torch, numpy, huggingface_hub and safetensors,
and neither transformers, timm nor torchaudio. Each member is fed its own preprocessing, as it was scored on AudioSet's
evaluation set (github.com/Scicom-AI-Enterprise-Organization/Melati, audioset/zoo.py), computed for the whole batch on
the device in fp32.

    import sys
    from huggingface_hub import snapshot_download

    path = snapshot_download("Scicom-intl/Melati-AudioSet-Ensemble")
    sys.path.insert(0, path)
    from melati_ensemble import Ensemble

    model = Ensemble(path)                  # on the GPU when there is one; bf16=True is several times faster
    probs = model([wav])                    # 16 kHz mono clips, each cut or zero-padded to 10 s -> (clips, 527)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR, CLIP = 16000, 160000        # every member was scored on 10 s clips, a shorter one padded with zeros
_CONST = {}


def _const(name, device, make):
    key = (name, str(device))
    if key not in _CONST:
        _CONST[key] = make().to(device)
    return _CONST[key]


# ------------------------------------------------------------------------------------------------ preprocessing
def _kaldi_banks():
    """torchaudio.compliance.kaldi.get_mel_banks(128, 512, 16000, 20, 0, 100, -500, 1.0), computed as torchaudio
    computes it, with the zero column kaldi.fbank adds for the Nyquist bin: (128, 257)."""
    bins, padded, sr, low = 128, 512, 16000.0, 20.0
    mel_low, mel_high = 1127.0 * math.log(1.0 + low / 700.0), 1127.0 * math.log(1.0 + 0.5 * sr / 700.0)
    delta = (mel_high - mel_low) / (bins + 1)
    b = torch.arange(bins).unsqueeze(1)
    left, center, right = mel_low + b * delta, mel_low + (b + 1.0) * delta, mel_low + (b + 2.0) * delta
    mel = (1127.0 * (1.0 + (sr / padded) * torch.arange(padded / 2) / 700.0).log()).unsqueeze(0)
    banks = torch.max(torch.zeros(1), torch.min((mel - left) / (center - left), (right - mel) / (right - center)))
    return F.pad(banks, (0, 1))


def kaldi_fbank(w, window="hanning"):
    """torchaudio.compliance.kaldi.fbank as EAT, SSLAM and BEATs call it (128 bins, 25 ms windows every 10 ms, DC
    removed per frame, pre-emphasis 0.97, no dither, log mel power floored at float eps; EAT's Hanning window, or
    BEATs's, Kaldi's default "povey"), for a batch at once: (B, samples) at 16 kHz -> (B, frames, 128). The library
    does one waveform a call."""
    banks = _const("kaldi_banks", w.device, _kaldi_banks)
    win = _const(f"kaldi_{window}", w.device,
                 lambda: torch.hann_window(400, periodic=False).pow(0.85) if window == "povey"
                 else torch.hann_window(400, periodic=False))
    fr = w.float().unfold(1, 400, 160)                                         # snip_edges: whole frames only
    fr = fr - fr.mean(-1, keepdim=True)
    fr = fr - 0.97 * torch.cat([fr[..., :1], fr[..., :-1]], -1)
    spec = torch.fft.rfft(fr * win, n=512).abs().pow(2)                         # (B, frames, 257)
    return torch.clamp(spec @ banks.T, min=torch.finfo(torch.float32).eps).log()


def _ced_banks():
    """torchaudio.functional.melscale_fbanks(257, 0, 8000, 64, 16000), HTK without norm, as CED-base's feature
    extractor builds it: (257, 64)."""
    freqs = torch.linspace(0, 16000 // 2, 257)
    pts = 700.0 * (10.0 ** (torch.linspace(0.0, 2595.0 * math.log10(1.0 + 8000 / 700.0), 66) / 2595.0) - 1.0)
    diff, slopes = pts[1:] - pts[:-1], pts.unsqueeze(0) - freqs.unsqueeze(1)
    return torch.max(torch.zeros(1), torch.min((-1.0 * slopes[:, :-2]) / diff[:-1], slopes[:, 2:] / diff[1:]))


def ced_mel(w):
    """CED-base's feature extractor (a 512-point Hann STFT every 160 samples, centred with reflection, power, 64 mel
    bins to 8 kHz, in decibels floored 120 dB under the loudest value) for a batch at once, the floor taken under each
    clip's own loudest value, which is what the extractor gives a clip alone: (B, samples) -> (B, 64, frames). The
    extractor floors under the loudest value in everything it is given at once, so there a clip's tags would depend on
    the clips batched with it."""
    banks = _const("ced_banks", w.device, _ced_banks)
    win = _const("ced_window", w.device, lambda: torch.hann_window(512))
    spec = torch.stft(w.float(), 512, hop_length=160, win_length=512, window=win, center=True, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=True).abs().pow(2.0)
    db = 10.0 * torch.log10(torch.clamp(torch.matmul(spec.transpose(-1, -2), banks).transpose(-1, -2), min=1e-10))
    return torch.max(db, db.amax(dim=(-2, -1), keepdim=True) - 120.0)


# ------------------------------------------------------------------------------------------------ the networks
class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    """A transformer block, normalised after each sublayer (EAT's, data2vec 2.0's with layer_norm_first off) or before
    it (CED's)."""

    def __init__(self, dim, heads, eps, post):
        super().__init__()
        self.post = post
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)
        self.attn, self.mlp = _Attention(dim, heads), _Mlp(dim, dim * 4)

    def forward(self, x):
        if self.post:
            r = self.norm1(x + self.attn(x))
            return self.norm2(r + self.mlp(r))
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _sincos(dim, rows, cols):
    """EAT's fixed 2-D sin-cos positions (get_2d_sincos_pos_embed_flexible), rows of patches in time, cols in
    frequency: (1, rows * cols, dim)."""
    def one(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float32) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], 1)
    grid = np.stack(np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32)), 0)
    return torch.from_numpy(np.concatenate([one(dim // 2, grid[0]), one(dim // 2, grid[1])], 1)).float()[None]


class EAT(nn.Module):
    """EAT (Chen et al., IJCAI 2024) fine-tuned on AudioSet, as EAT-large, SSLAM and Melati-AudioSet-EAT-Large hold it:
    the modeling code of worstchan/EAT-large_epoch20_finetune_AS2M without its dropout, its parameters named as there
    so model.safetensors loads as it is. (B, 1, 1024, 128) normalised filterbanks -> (B, 527) logits."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.get("model_variant") == "finetune" and not cfg.get("layer_norm_first"), "not a fine-tuned EAT"
        D, P = cfg["embed_dim"], cfg["patch_size"]
        self.local_encoder = nn.Module()
        self.local_encoder.proj = nn.Conv2d(1, D, P, cfg["stride"])
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, D))
        self.fixed_positional_encoder = nn.Module()
        self.fixed_positional_encoder.positions = nn.Parameter(
            _sincos(D, cfg["max_length"], cfg["mel_bins"] // P), requires_grad=False)
        self.pre_norm = nn.LayerNorm(D, eps=cfg["norm_eps"])
        self.blocks = nn.ModuleList(_Block(D, cfg["num_heads"], cfg["norm_eps"], post=True)
                                    for _ in range(cfg["depth"]))
        self.fc_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg["num_classes"])

    def forward(self, x):
        x = self.local_encoder.proj(x).flatten(2).transpose(1, 2)
        x = x + self.fixed_positional_encoder.positions[:, :x.size(1)]
        x = self.pre_norm(torch.cat([self.extra_tokens.expand(len(x), -1, -1), x], 1))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.fc_norm(x[:, 0]))


class CED(nn.Module):
    """CED (Dinkel et al., ICASSP 2024; Apache-2.0) fine-tuned on AudioSet, as mispeech/ced-base holds it: its modeling
    code for one chunk (10 s is 1001 of its 1012 frames) without dropout, its parameters named as there so
    model.safetensors loads as it is. (B, 64, frames) decibels -> (B, 527) probabilities (mean pooling) or logits."""

    def __init__(self, cfg):
        super().__init__()
        D, S = cfg["embed_dim"], cfg["patch_stride"]
        self.frames, self.pooling = cfg["target_length"], cfg.get("pooling", "mean")
        assert self.pooling in ("mean", "logit"), self.pooling
        enc = self.encoder = nn.Module()
        enc.init_bn = nn.BatchNorm2d(cfg["n_mels"], momentum=0.01)
        enc.patch_embed = nn.Module()
        enc.patch_embed.proj = nn.Conv2d(1, D, cfg["patch_size"], S)
        enc.time_pos_embed = nn.Parameter(torch.zeros(1, D, 1, self.frames // S))
        enc.freq_pos_embed = nn.Parameter(torch.zeros(1, D, cfg["n_mels"] // S, 1))
        enc.blocks = nn.Sequential(*[_Block(D, cfg["num_heads"], 1e-6, post=False) for _ in range(cfg["depth"])])
        enc.norm = nn.LayerNorm(D, eps=1e-6)
        self.outputlayer = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, cfg["outputdim"]))

    def forward(self, mel):
        assert mel.shape[-1] <= self.frames, f"{mel.shape[-1]} frames: longer than one chunk of {self.frames}"
        enc = self.encoder
        x = enc.init_bn(mel[:, None].transpose(1, 2)).transpose(1, 2)          # batch norm over the mel bins
        x = enc.patch_embed.proj(x)
        x = x + enc.time_pos_embed[..., :x.shape[-1]] + enc.freq_pos_embed
        x = enc.norm(enc.blocks(x.flatten(2, 3).permute(0, 2, 1)))
        out = self.outputlayer(x.mean(1))
        return torch.sigmoid(out) if self.pooling == "mean" else out


# ------------------------------------------------------------------------------------------------ members
def _hub(repo, revision):
    """A member's config.json and weights from its own repo at the pinned revision: the only files fetched."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    return cfg, load_file(hf_hub_download(repo, "model.safetensors", revision=revision))


def _load(net, state, prefix=""):
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "fixed_positional_encoder.positions"         # rebuilt above if not saved
               and not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(f"weights do not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return net.eval()


class _EATMember:
    """EAT-large, SSLAM and Melati-AudioSet-EAT-Large: Kaldi filterbanks (Hanning), 1024 frames (10.24 s), normalised as
    EAT's card says; logits, through a sigmoid. Their outputs are in AudioSet's order."""
    feature = "eat"

    def __init__(self, repo, revision, labels):
        cfg, state = _hub(repo, revision)
        names = [str(v) for _, v in sorted((int(k), v) for k, v in (cfg.get("id2label") or {}).items())]
        assert not names or names[0].startswith("LABEL_") or names == labels, f"{repo}: not in AudioSet's order"
        self.net = _load(EAT(cfg), state, "model.")

    def __call__(self, feats):
        return torch.sigmoid(self.net(feats["eat"]).float())


class _CEDMember:
    """CED-base: its own 64-bin mel spectrogram in decibels (ced_mel); with mean pooling it gives probabilities. Its
    outputs are in AudioSet's order (its config's own names are shortened ones, so they are not compared)."""
    feature = "ced"

    def __init__(self, repo, revision):
        cfg, state = _hub(repo, revision)
        cfg.setdefault("n_mels", 64)
        self.net = _load(CED(cfg), state)

    def __call__(self, feats):
        out = self.net(feats["ced"]).float()
        return out if self.net.pooling == "mean" else torch.sigmoid(out)


class _BEATsMember:
    """BEATs fine-tuned on AudioSet (iter3+, AS2M, cpt2): its filterbanks (Kaldi's Povey window, the waveform scaled to
    16-bit range, normalised as its code does) computed here for the batch, then its own patch embedding, encoder and
    predictor, mean over frames, sigmoid; in the order of the labels its checkpoint carries (beats.json gives AudioSet's
    index for each)."""
    feature = "beats"

    def __init__(self, path):
        from safetensors.torch import load_file
        from melati_beats.BEATs import BEATs, BEATsConfig
        meta = json.loads((path / "beats.json").read_text())
        self.net = BEATs(BEATsConfig(meta["cfg"]))
        self.net.load_state_dict(load_file(str(path / "beats.safetensors")))
        self.net.eval()
        self.order = torch.as_tensor(meta["order"])
        for mod in self.net.modules():
            if getattr(mod, "has_relative_attention_bias", False):
                mod.compute_bias = _cached_bias(mod)

    def __call__(self, feats):
        m = self.net
        x = m.patch_embedding(feats["beats"].unsqueeze(1))
        x = m.layer_norm(x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2))
        if m.post_extract_proj is not None:
            x = m.post_extract_proj(x)
        x, _ = m.encoder(x, padding_mask=None)
        p = torch.sigmoid(m.predictor(x).mean(1)).float()
        if self.order.device != p.device:                                       # moved once, not on every call
            self.order = self.order.to(p.device)
        q = torch.empty_like(p)
        q[:, self.order] = p                                                    # into AudioSet's order
        return q


def _cached_bias(attn):
    """BEATs's MultiheadAttention.compute_bias with its bucket indices built once and kept on the device: unilm's builds
    them on the CPU on every call and copies them over, which a CUDA graph cannot hold. The same indices, the same
    embedding lookup."""
    buckets = {}

    def compute_bias(query_length, key_length):
        dev = attn.relative_attention_bias.weight.device
        if (query_length, key_length, dev) not in buckets:
            rel = torch.arange(key_length, dtype=torch.long)[None, :] - torch.arange(query_length, dtype=torch.long)[:, None]
            buckets[query_length, key_length, dev] = attn._relative_positions_bucket(rel, bidirectional=True).to(dev)
        return attn.relative_attention_bias(buckets[query_length, key_length, dev]).permute([2, 0, 1])
    return compute_bias


def features(w, kinds):
    """The preprocessing each kind of member needs, once for the batch, in fp32: (B, 160000) -> {kind: tensor}."""
    out = {}
    if "eat" in kinds:
        f = kaldi_fbank(w - w.mean(1, keepdim=True))
        f = F.pad(f, (0, 0, 0, max(0, 1024 - f.shape[1])))[:, :1024]
        out["eat"] = ((f + 4.268) / (4.569 * 2))[:, None]
    if "beats" in kinds:
        out["beats"] = (kaldi_fbank(w * 2 ** 15, "povey") - 15.41663) / (2 * 6.55582)
    if "ced" in kinds:
        out["ced"] = ced_mel(w)
    return out


def resample(x, sr, new=SR):
    """torchaudio.functional.resample's default (windowed sinc, a Hann window over 6 zero crossings, rolloff 0.99),
    ported (torchaudio is BSD-2-Clause) so that reading a file needs no torchaudio: (..., samples) float tensor."""
    if sr == new:
        return x
    g = math.gcd(int(sr), int(new))
    orig, new = int(sr) // g, int(new) // g
    base = min(orig, new) * 0.99
    width = math.ceil(6 * orig / base)
    idx = torch.arange(-width, width + orig, dtype=x.dtype)[None, None] / orig
    t = (torch.arange(0, -new, -1, dtype=x.dtype)[:, None, None] / new + idx) * base
    t = t.clamp_(-6, 6)
    window = torch.cos(t * math.pi / 6 / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t) * (window * (base / orig))
    shape, n = x.shape, x.shape[-1]
    y = F.conv1d(F.pad(x.reshape(-1, n), (width, width + orig))[:, None], kernel.to(x.device), stride=orig)
    y = y.transpose(1, 2).reshape(y.shape[0], -1)[..., :int(torch.ceil(torch.as_tensor(new * n / orig)))]
    return y.reshape(shape[:-1] + y.shape[-1:])


def read(f):
    """An audio file (a path or a file object: WAV, FLAC, OGG or MP3, anything soundfile reads), its channels
    averaged, at 16 kHz -> float32 numpy samples."""
    import soundfile as sf
    x, sr = sf.read(f, dtype="float32", always_2d=True)
    return resample(torch.from_numpy(x.mean(1)), sr).numpy()


def _ten_s(x):
    x = x.detach().float().cpu() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, np.float32))
    if x.dim() != 1:
        raise ValueError(f"a clip is one channel of 16 kHz samples, not an array of shape {tuple(x.shape)}")
    x = x[:CLIP]
    return F.pad(x, (0, CLIP - len(x)))


class Ensemble:
    """The members' probabilities, each put in AudioSet's class order (self.labels), averaged.

    path     this repo, downloaded (huggingface_hub.snapshot_download)
    members  the members to run (default: all six, ensemble.json's names); the reported mAP is all six's
    device   default: the GPU when there is one
    bf16     the networks in bf16 autocast on the GPU (the preprocessing stays fp32), several times faster; the card
             gives its mAP. The reported mAP is fp32's
    """

    def __init__(self, path, members=None, device=None, bf16=False):
        self.path = Path(path)
        if str(self.path) not in sys.path:                 # melati_beats, BEATs's code, sits beside this file
            sys.path.insert(0, str(self.path))
        spec = json.loads((self.path / "ensemble.json").read_text())
        self.labels = [c["name"] for c in json.loads((self.path / "labels.json").read_text())]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        names = [m["name"] for m in spec["members"]]
        unknown = [m for m in (members or []) if m not in names]
        if unknown:
            raise ValueError(f"no member {unknown}: the members are {names}")
        self.members = {}
        for m in spec["members"]:
            if members and m["name"] not in members:
                continue
            if m["kind"] == "beats":
                member = _BEATsMember(self.path)
            elif m["kind"] == "ced":
                member = _CEDMember(m["repo"], m["revision"])
            else:
                member = _EATMember(m["repo"], m["revision"], self.labels)
            member.net.to(self.device)
            self.members[m["name"]] = member
        self.bf16 = bf16

    def bf16_weights(self):
        """Hold every Linear and Conv2d weight in bf16, for good: what bf16 autocast casts them to on every call, cast
        once (melati_engine.py serves this way). fp32 is then no longer available from this object."""
        self.bf16 = True
        for member in self.members.values():
            for mod in member.net.modules():
                if type(mod) in (nn.Linear, nn.Conv2d):
                    mod.to(torch.bfloat16)
        return self

    @torch.no_grad()
    def forward(self, w):
        """(B, 160000) fp32 on self.device -> {member: (B, 527) fp32 probabilities in AudioSet's order}."""
        feats = features(w, {m.feature for m in self.members.values()})
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16 and self.device.type == "cuda"):
            return {k: m(feats) for k, m in self.members.items()}

    @torch.no_grad()
    def __call__(self, clips, batch=16, each=False):
        """16 kHz mono clips (arrays or tensors), each cut or zero-padded to 10 s -> (clips, 527) probabilities, the
        members' average; with each=True, also {member: its (clips, 527) probabilities}."""
        if getattr(clips, "ndim", None) == 1:              # one clip given bare
            clips = [clips]
        out = {k: [] for k in self.members}
        for i in range(0, len(clips), batch):
            w = torch.stack([_ten_s(x) for x in clips[i:i + batch]]).to(self.device)
            for k, p in self.forward(w).items():
                out[k].append(p.cpu().numpy())
        per = {k: np.concatenate(v) for k, v in out.items()}
        avg = np.mean(list(per.values()), 0)
        return (avg, per) if each else avg

    def windows(self, wave, hop_s=5.0, batch=16):
        """A longer recording (16 kHz mono) as 10 s windows every hop_s seconds, the last one ending where it ends ->
        (each window's start in seconds, (windows, 527) probabilities). The highest over the windows tags the whole."""
        x = np.asarray(wave.detach().float().cpu() if torch.is_tensor(wave) else wave, np.float32)
        hop = int(round(hop_s * SR))
        starts = list(range(0, max(len(x) - CLIP, 0) + 1, hop))
        if len(x) > CLIP and starts[-1] + CLIP < len(x):
            starts.append(len(x) - CLIP)
        return np.array(starts) / SR, self([x[s:s + CLIP] for s in starts], batch=batch)
