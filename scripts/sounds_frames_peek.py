#!/usr/bin/env python3
"""A few segments' Music frames from the sounds stage, as text: what the inline tag's onset rests on. RUNS ON THE BOX.

    python scripts/sounds_frames_peek.py r1adb558d8448_0000 r00c843e9ff84_0002
"""
import os
import json
import sys
from pathlib import Path

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

for seg in sys.argv[1:]:
    rec = seg.rsplit("_", 1)[0]
    row = next(json.loads(line) for line in (Path(f"{MELATI_DATA}/podcast/work") / rec / "sounds.jsonl").open()
               if json.loads(line)["seg"] == seg)
    p = row["frames"]["Music"]
    print(f"{seg}: v4 Music {row['p']['Music']}, {len(p)} steps of {row['frames']['step_s']} s")
    print("  " + " ".join(f"{v:.2f}" for v in p))
