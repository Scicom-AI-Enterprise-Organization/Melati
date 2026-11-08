"""The clip half of a frame model (audioset/frames_train.py saves the encoder and both heads): its EAT weights alone,
as a checkpoint audioset/zoo.py and audioset/release.py read like any audioset/train.py one, keeping its lineage.

    python audioset/clip_of_frames.py $MELATI_DATA/models/audioset/frames1/best.pt
"""
import sys
from pathlib import Path

import torch

src = Path(sys.argv[1])
ck = torch.load(src, map_location="cpu", weights_only=False)
state = {k[len("eat."):]: v for k, v in ck["state"].items() if k.startswith("eat.")}
out = src.with_name("clip.pt")
torch.save({"state": state, "init": ck["init"], "from": ck["from"], "map": ck["clip_map"], "frame_map": ck["frame_map"],
            "step": ck["step"], "frames": str(src)}, out)
print(f"{out}: {len(state)} tensors, clip mAP {ck['clip_map']:.4f} (bf16), frame mAP {ck['frame_map']:.4f}")
