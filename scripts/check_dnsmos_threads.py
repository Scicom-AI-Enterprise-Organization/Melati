#!/usr/bin/env python3
"""DNSMOS with one thread a call (melati.third_party.dnsmos, as the tag server runs it) against ONNX Runtime's
default session, on real segments of a finished run: the scores must not move. CPU only. RUNS ON THE BOX.

    python scripts/check_dnsmos_threads.py --work $MELATI_DATA/podcast/work --clips 40
"""
import os
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati.third_party.dnsmos import ComputeScore  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--clips", type=int, default=40)
ap.add_argument("--onnx", default=f"{MELATI_DATA}/models/dnsmos/sig_bak_ovr.onnx")
args = ap.parse_args()
clips = []
for d in sorted(args.work.iterdir()):
    if not (d / "fill.jsonl").exists():
        continue
    x, _ = sf.read(str(d / "vocal16k.flac"), dtype="float32")
    for l in (d / "segments.jsonl").open():
        s = json.loads(l)
        clips.append(x[int(s["start"] * 16000):int(s["end"] * 16000)])
    if len(clips) >= args.clips:
        break
import onnxruntime as ort  # noqa: E402

one = ComputeScore(args.onnx, "cpu")
default = ComputeScore(args.onnx, "cpu")
default.onnx_sess = ort.InferenceSession(args.onnx)
diff = 0.0
for c in clips[: args.clips]:
    a, b = one(c, 16000, False), default(c, 16000, False)
    diff = max(diff, *(abs(float(a[k]) - float(b[k])) for k in ("OVRL", "SIG", "BAK")))
print(f"{min(len(clips), args.clips)} clips, max |score difference| {diff:.2e}")
print("DNSMOS_SAME" if diff < 1e-4 else "DNSMOS_DIFFERS")
