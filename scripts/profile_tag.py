#!/usr/bin/env python3
"""What the w2v-BERT and WavLM heads of the tag server spend on the CPU: their feature extractors on real segments,
one clip after another (as melati.ser.hf_ssl runs them) against a thread pool, and whether the pool's features are
identical. CPU only. RUNS ON THE BOX.

    python scripts/profile_tag.py --work $MELATI_DATA/podcast/work --clips 200
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--clips", type=int, default=200)
ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16])
args = ap.parse_args()
clips = []
for d in sorted(args.work.iterdir()):
    if not (d / "fill.jsonl").exists():
        continue
    x, _ = sf.read(str(d / "vocal16k.flac"), dtype="float32")
    for l in (d / "segments.jsonl").open():
        s = json.loads(l)
        c = x[int(s["start"] * 16000):int(s["end"] * 16000)][: 30 * 16000]
        if len(c) >= 4800:
            clips.append(c)
    if len(clips) >= args.clips:
        break
clips = clips[: args.clips]
print(f"{len(clips)} clips, {sum(map(len, clips)) / 16000 / 60:.1f} min of audio")
from transformers import AutoFeatureExtractor  # noqa: E402

for name in ("facebook/w2v-bert-2.0", "microsoft/wavlm-large"):
    fe = AutoFeatureExtractor.from_pretrained(name)
    one = lambda x: fe(x, sampling_rate=16000, return_tensors="np")  # noqa: E731
    t = time.perf_counter()
    ref = [one(x) for x in clips]
    t1 = time.perf_counter() - t
    print(f"{name}: one at a time {t1:.1f} s ({1000 * t1 / len(clips):.0f} ms a clip)")
    key = next(iter(ref[0].keys()))
    for k in args.threads:
        with ThreadPoolExecutor(k) as ex:
            t = time.perf_counter()
            out = list(ex.map(one, clips))
            tk = time.perf_counter() - t
        same = all(np.array_equal(a[key], b[key]) for a, b in zip(ref, out))
        print(f"  {k:2d} threads: {tk:.1f} s, {t1 / tk:.1f}x, identical features: {same}")
