"""A Melati classifier in one checkpoint: the Whisper-large-v3 encoder and the head trained on its frozen features.

Both of Melati's released classifiers (non-verbal sounds, emotion) are this model; the head is what Melati's pipeline
runs (melati/ser.py): every encoder layer (and the embeddings) mean-pooled over the clip's own frames, a learned
softmax mix of the layers, LayerNorm, one hidden layer with GELU, and one output per label. Whisper pads every input to
30 s, so pass `num_frames` (50 per second of the clip) or the padding is averaged in. The pooled features were stored
as float16 when the head was trained, so they are rounded the same way here.

A config with `frame_head` also carries Melati's frame-level detector (melati/nv_frames.py) on the same encoder:
`model.frames(input_features)` gives, every 1/fps s of the 30 s input, an independent probability per class
(laughter and coughs over speech too, which no word gap holds).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, WhisperConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

try:
    from .configuration_melati_head import MelatiHeadConfig
except ImportError:                                    # imported as a plain module, outside a package
    from configuration_melati_head import MelatiHeadConfig


class MelatiFrameHead(nn.Module):
    """A softmax mix of the chosen encoder layers, LayerNorm, a projection, then residual dilated convolutions over
    time (melati/nv_frames.py's FrameHead, without its dropout)."""

    def __init__(self, layers, dim, hidden, n, kernel, dilations):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(layers))
        self.norm = nn.LayerNorm(dim)
        self.inp = nn.Linear(dim, hidden)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in dilations)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel, padding=d * (kernel - 1) // 2, dilation=d)
                                   for d in dilations)
        self.out = nn.Linear(hidden, n)

    def forward(self, x):                                   # B, T, L, D -> B, T, n logits
        h = (self.mix.float().softmax(0)[None, None, :, None] * x).sum(2)
        h = F.gelu(self.inp(self.norm(h)))
        for nm, conv in zip(self.norms, self.convs):
            h = h + F.gelu(conv(nm(h).transpose(1, 2)).transpose(1, 2))
        return self.out(h)


class MelatiHeadModel(PreTrainedModel):
    config_class = MelatiHeadConfig
    main_input_name = "input_features"
    # the head runs in float32, as it was trained; its names must not match any encoder module (Whisper's layers
    # have their own fc1 / fc2, which would then stay float32 inside a float16 encoder)
    _keep_in_fp32_modules = ["layer_mix", "head_norm", "head_fc1", "head_out", "frame_head"]

    def __init__(self, config):
        super().__init__(config)
        enc = WhisperConfig(**config.encoder)
        self.encoder = WhisperEncoder(enc)
        self.layer_mix = nn.Parameter(torch.zeros(enc.encoder_layers + 1))
        self.head_norm = nn.LayerNorm(enc.d_model)
        self.head_fc1 = nn.Linear(enc.d_model, config.hidden)
        self.head_out = nn.Linear(config.hidden, len(config.labels))
        fh = getattr(config, "frame_head", None) or {}
        if fh:
            self.frame_head = MelatiFrameHead(len(fh["layers"]), enc.d_model, fh["hidden"], len(fh["classes"]),
                                              fh["kernel"], fh["dilations"])
        self.post_init()

    def _init_weights(self, module):
        pass                                       # every weight comes from the checkpoint

    def pooled(self, input_features, num_frames=None):
        """(B, layers, dim): each encoder layer averaged over the first num_frames frames of its clip."""
        enc_dtype = self.encoder.conv1.weight.dtype
        h = torch.stack(self.encoder(input_features.to(enc_dtype), output_hidden_states=True).hidden_states, 1)
        t = h.shape[2]
        if num_frames is None:
            num_frames = torch.full((h.shape[0],), t)
        n = torch.as_tensor(num_frames, device=h.device).clamp(1, t)
        mask = (torch.arange(t, device=h.device)[None, :] < n[:, None]).float()          # B, T
        x = (h.float() * mask[:, None, :, None]).sum(2) / n[:, None, None].float()
        return x.half().float()

    def frames(self, input_features):
        """(B, 30 * fps, classes) probabilities over the whole 30 s input (frames past the clip are padding)."""
        fh = self.config.frame_head
        enc_dtype = self.encoder.conv1.weight.dtype
        hs = self.encoder(input_features.to(enc_dtype), output_hidden_states=True).hidden_states
        h = torch.stack([hs[k] for k in fh["layers"]], 2)                 # B, 1500, L, D, in the encoder's dtype
        t = int(30 * fh["fps"])
        h = h.view(h.shape[0], t, h.shape[1] // t, *h.shape[2:]).mean(2)  # averaged as the training features were
        return torch.sigmoid(self.frame_head(h.float()))

    def forward(self, input_features, num_frames=None, labels=None, **kwargs):
        x = self.pooled(input_features, num_frames)
        z = (self.layer_mix.float().softmax(0)[None, :, None] * x).sum(1)
        z = F.layer_norm(z, (z.shape[-1],), self.head_norm.weight.float(), self.head_norm.bias.float(),
                         self.head_norm.eps)
        z = F.gelu(F.linear(z, self.head_fc1.weight.float(), self.head_fc1.bias.float()))
        logits = F.linear(z, self.head_out.weight.float(), self.head_out.bias.float())
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
