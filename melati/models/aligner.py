"""Where-Words-Begin-and-End as a batch function, with a confidence per word (serve/align_server.py).

The aligner (Scicom-intl/Where-Words-Begin-and-End-Whisper-Large-V3) reads the transcript once,
teacher-forced, with two slot tokens after each word, and scores each slot's hidden state against
Whisper's 1,501 timestamp tokens; a word's time is the expectation over the tokens around the most
likely one. 10.7 ms mean boundary error on TIMIT, 16.0 ms on real landline calls.

Each word also gets `c`: the probability mass the model puts on the window around its chosen
timestamp (the lower of the start and end slots). A word whose time the model is unsure of is a
word the transcript may have wrong, so downstream reads it as alignment confidence.

The model is loaded through the release's own `align.py` (shipped in the model repository); the
batching and the confidence are added here.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

REPO = "Scicom-intl/Where-Words-Begin-and-End-Whisper-Large-V3"
SR = 16000


class Aligner:
    def __init__(self, repo: str = REPO, device: str = "cuda"):
        from huggingface_hub import snapshot_download
        from melati import pins
        path = snapshot_download(repo, revision=pins.rev(repo))
        sys.path.insert(0, path)
        # The release was saved with transformers 5 and the aligner service runs on it
        # (deploy/box_align_setup.sh). On transformers 4 the repository's tokenizer.json-only layout
        # makes the slow WhisperTokenizer `open(None)`, so there the fast one is swapped in.
        import transformers
        slow = transformers.WhisperTokenizer
        if int(transformers.__version__.split(".")[0]) < 5:
            transformers.WhisperTokenizer = transformers.WhisperTokenizerFast
        try:
            from align import N_TS, ForcedAligner, words_of   # the release's own code
            self.fa = ForcedAligner(path, device=device)
        finally:
            transformers.WhisperTokenizer = slow
        self.words_of, self.N_TS = words_of, N_TS

    @torch.no_grad()
    def __call__(self, items):
        """[(audio 16 kHz float32 <= 30 s, transcript, language or None)] -> [{"words", "score"} or ValueError]."""
        fa, N_TS = self.fa, self.N_TS
        out, seqs, keep, units = [None] * len(items), [], [], []
        for i, (x, t, l) in enumerate(items):
            words = self.words_of(t)
            ids, slots = fa._sequence(words, l)
            if not words or len(ids) > 448 or len(x) > 30 * SR:
                out[i] = ValueError(f"does not fit one window ({len(ids)} tokens, {len(x) / SR:.1f} s)")
                continue
            seqs.append((ids, slots)); keep.append(i); units.append(words)
        if not keep:
            return out
        feats = fa.fe([np.asarray(items[i][0], dtype=np.float32) for i in keep], sampling_rate=SR,
                      return_tensors="pt", device=fa.device)["input_features"].to(fa.device, fa.dtype)
        L = max(len(s) for s, _ in seqs)
        ids = torch.full((len(seqs), L), fa.eot, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        rows = []
        for b, (s, slots) in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s)
            attn[b, :len(s)] = 1
            rows += [(b, p) for p in slots]
        h = fa.model.model(input_features=feats, decoder_input_ids=ids.to(fa.device),
                           decoder_attention_mask=attn.to(fa.device)).last_hidden_state
        r = torch.tensor(rows, device=fa.device)
        ts_w = fa.model.model.decoder.embed_tokens.weight[fa.ts0: fa.ts0 + N_TS]
        logits = h[r[:, 0], r[:, 1]].float() @ ts_w.float().T
        t = fa._times(logits)
        p = logits.softmax(-1)
        k = p.argmax(-1)
        idx = (k[:, None] + torch.arange(-fa.radius, fa.radius + 1, device=p.device)[None]).clamp(0, N_TS - 1)
        mass = p.gather(1, idx).sum(1).cpu().numpy()
        j = 0
        for b, i in enumerate(keep):
            n = len(units[b])
            tt, mm = t[j: j + 2 * n].reshape(n, 2), mass[j: j + 2 * n].reshape(n, 2)
            j += 2 * n
            dur, prev, res = len(items[i][0]) / SR, 0.0, []
            for q, w in enumerate(units[b]):
                s = max(float(tt[q, 0]), prev)
                e = max(float(tt[q, 1]), s + 0.01)
                prev = min(s, dur)
                res.append({"w": w, "s": round(min(s, dur), 3), "e": round(min(e, dur), 3),
                            "c": round(float(min(mm[q, 0], mm[q, 1])), 3)})
            out[i] = {"words": res, "score": round(float(np.mean([w["c"] for w in res])), 4)}
        return out
