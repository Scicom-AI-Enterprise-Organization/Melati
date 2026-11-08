#!/usr/bin/env python3
"""melati.ser.hf_ssl batched against one clip at a time, on the clips the tag stage sends (segments and the phrases
between their pauses, from a finished run): the frames each clip is pooled over, the pooled features, the emotion
heads' probabilities and their argmax, and the time each way takes. RUNS ON THE BOX (one GPU).

    CUDA_VISIBLE_DEVICES=4 python scripts/check_ssl_batching.py --work $MELATI_DATA/podcast/work --clips 400
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import emotion as EMO  # noqa: E402
from melati import ser as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--clips", type=int, default=400)
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()
clips = []
for d in sorted(args.work.iterdir()):
    if not (d / "fill.jsonl").exists():
        continue
    x, _ = sf.read(str(d / "vocal16k.flac"), dtype="float32")
    al = {json.loads(l)["seg"]: json.loads(l) for l in (d / "align.jsonl").open()}
    for l in (d / "segments.jsonl").open():
        s = json.loads(l)
        seg = x[int(s["start"] * S.SR):int(s["end"] * S.SR)]
        if len(seg) >= int(0.3 * S.SR):
            clips.append(seg)
        for a, b, ps, pe in EMO.phrases((al.get(s["seg"]) or {}).get("words") or []):
            c = seg[int(ps * S.SR):int(pe * S.SR)]
            if len(c) >= int(0.3 * S.SR):
                clips.append(c)
    if len(clips) >= args.clips:
        break
clips = S.prepare(clips[: args.clips])
lens = np.array([len(c) / S.SR for c in clips])
print(f"{len(clips)} clips, {lens.sum() / 60:.1f} min, median {np.median(lens):.1f} s, max {lens.max():.1f} s")
import torch  # noqa: E402
from transformers import AutoFeatureExtractor  # noqa: E402

fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")      # hf_ssl's frame count, w2v-BERT's own
alone = [fe(c, sampling_rate=S.SR, return_tensors="np")["input_features"].shape[1] for c in clips]
assert all(k == (1 + (len(c) - 400) // 160 + 1) // 2 for k, c in zip(alone, clips)), "w2v-BERT frame count"
res = {}
for enc, name in (("w2vbert", "facebook/w2v-bert-2.0"), ("wavlm", "microsoft/wavlm-large")):
    one, many = S.hf_ssl(name, "cuda", batched=False), S.hf_ssl(name, "cuda", batched=True)
    with torch.inference_mode():
        t = time.perf_counter(); a = np.concatenate([one(clips[i:i + 32]) for i in range(0, len(clips), 32)])
        t1 = time.perf_counter() - t
        t = time.perf_counter(); b = np.concatenate([many(clips[i:i + 32]) for i in range(0, len(clips), 32)])
        t2 = time.perf_counter() - t
    ck = torch.load(os.path.join(S.HEADS_DIR, f"{enc}.pt"), map_location="cpu")
    head = S.head_module(ck["layers"], ck["dim"]).cuda().eval()
    head.load_state_dict(ck["state"])
    with torch.inference_mode():
        pa, pb = (head(torch.from_numpy(v.astype(np.float16)).cuda().float()).softmax(-1).cpu().numpy() for v in (a, b))
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    rel = np.abs(a - b).max() / np.abs(a).max()
    r = {"one_at_a_time_s": round(t1, 1), "batched_s": round(t2, 1), "speedup": round(t1 / t2, 2),
         "feature_max_rel_diff": float(f"{rel:.2e}"), "feature_min_cosine": float(f"{cos.min():.6f}"),
         "prob_max_abs_diff": float(f"{np.abs(pa - pb).max():.2e}"), "argmax_agree": float((pa.argmax(1) == pb.argmax(1)).mean())}
    res[enc] = r
    print(enc, json.dumps(r), flush=True)
    del one, many
    torch.cuda.empty_cache()
if args.out:
    args.out.write_text(json.dumps({"clips": len(clips), "minutes": round(lens.sum() / 60, 1), "encoders": res}, indent=1))
