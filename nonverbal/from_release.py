#!/usr/bin/env python3
"""The pipeline's two non-verbal heads from their public release, so the pipeline tags non-verbal sounds without
training them again.

Scicom-intl/Melati-NonVerbal-Whisper-Large-V3 is one checkpoint: the encoder of openai/whisper-large-v3 and both heads,
for AutoModel (nonverbal/release.py built it from the pipeline's own files with melati_hf/merge.py). The pipeline reads
each head from a file of its own and runs it on the encoder it loads from openai/whisper-large-v3 (melati.ser.HeadNV,
melati.nv_frames.FrameNV), so this writes the two files back from the release, which is 1.3 GB, most of it the encoder:

  $MELATI_DATA/models/nv_heads/whisper.pt    the gap head, nv_whisper: what a sound in a word gap is (breaths)
  $MELATI_DATA/models/nv_frames/whisper.pt   the frame head, nvf_whisper: laughter and coughs every 0.1 s

`--check` then runs both files the way the tag server does and the release the way its card does, on real speech, and
stops unless they agree as nonverbal/release.py required of the release (on a GPU):

    python nonverbal/from_release.py [--check <a parquet file with an audio column, or a folder of audio>]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from melati import nv_frames as NF  # noqa: E402
from melati import pins, ser  # noqa: E402
from melati.nv_gaps import HEAD_LABELS, clip_of  # noqa: E402

REPO = "Scicom-intl/Melati-NonVerbal-Whisper-Large-V3"
REVISION = pins.rev(REPO)
# the gap head's tensors in the release, by their names in the pipeline's head (melati.ser.head_module)
GAP = {"mix": "layer_mix", "net.0.weight": "head_norm.weight", "net.0.bias": "head_norm.bias",
       "net.1.weight": "head_fc1.weight", "net.1.bias": "head_fc1.bias", "net.4.weight": "head_out.weight",
       "net.4.bias": "head_out.bias"}


def save(ck, path: Path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save(ck, tmp)
    tmp.replace(path)
    print(f"{path}: {sum(v.numel() for v in ck['state'].values()):,} weights", flush=True)


def convert(snapshot: Path, revision: str, heads_dir: Path, frames_dir: Path):
    """Both heads out of the release, each in the file and layout the pipeline loads it from."""
    from safetensors.torch import load_file
    cfg = json.loads((snapshot / "config.json").read_text())
    st = load_file(str(snapshot / "model.safetensors"))
    assert tuple(cfg["labels"]) == HEAD_LABELS, cfg["labels"]
    gap = {k: st[v].float() for k, v in GAP.items()}
    layers, dim = gap["mix"].shape[0], gap["net.0.weight"].shape[0]
    ser.head_module(layers, dim, n=len(HEAD_LABELS)).load_state_dict(gap)       # the head HeadNV builds, strictly
    fh, head = cfg["frame_head"], NF.frame_head()                              # as FrameNV builds it
    assert (tuple(fh["classes"]), tuple(fh["layers"]), fh["fps"]) == (NF.CLASSES, NF.LAYERS, NF.FPS), fh
    built = (head.inp.out_features, head.convs[0].kernel_size[0], [c.dilation[0] for c in head.convs])
    assert built == (fh["hidden"], fh["kernel"], fh["dilations"]), (built, fh)
    frames = {k.removeprefix("frame_head."): v.float() for k, v in st.items() if k.startswith("frame_head.")}
    head.load_state_dict(frames)
    source = f"{REPO}@{revision}"
    save({"labels": list(HEAD_LABELS), "layers": int(layers), "dim": int(dim), "state": gap, "source": source},
         heads_dir / "whisper.pt")
    save({"classes": list(NF.CLASSES), "layers": list(NF.LAYERS), "fps": NF.FPS, "state": frames, "source": source},
         frames_dir / "whisper.pt")


def clips_from(src: Path, n: int):
    """Up to n recordings at 16 kHz: the `audio` column of a parquet file, as Hugging Face datasets keep audio, or the
    audio files in a folder."""
    from melati import audio as A
    if src.suffix == ".parquet":
        import pyarrow.parquet as pq
        cells = pq.read_table(src, columns=["audio"]).column("audio").to_pylist()[:n]
        return [A.decode(c["bytes"], sr=ser.SR)[0] for c in cells]
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"))[:n]
    return [A.decode(p, sr=ser.SR)[0] for p in files]


def check(snapshot: Path, heads_dir: Path, frames_dir: Path, src: Path, n: int):
    """The written files as the tag server runs them against the release as its card runs it: the gap head on cuts of
    0.3 to 2.4 s padded as a candidate is, the frame head on whole recordings and on one of over 30 s (read in
    windows). Probabilities within 5e-3 and the same top class, the test nonverbal/release.py held the release to."""
    sys.path.insert(0, str(snapshot))
    from melati_nonverbal import NonVerbal
    recs = [x[: int(28 * ser.SR)] for x in clips_from(src, n)]
    cuts = [clip_of(x, {"s": s, "e": s + 0.3 + (7 * s) % 2.1}) for x in recs
            for s in np.arange(0.2, len(x) / ser.SR - 2.5, 1.7)][:n]
    long = np.concatenate(recs)[: int(75 * ser.SR)]
    nv = NonVerbal(snapshot)
    a, b = nv.classify(cuts), ser.HeadNV("whisper", heads_dir=str(heads_dir))(cuts)[0]
    d, same = float(np.abs(a - b).max()), float((a.argmax(1) == b.argmax(1)).mean())
    fa = [nv.frames(x) for x in recs + [long]]
    fb = NF.FrameNV("whisper", models_dir=str(frames_dir))(recs + [long])[0]
    df = max(float(np.abs(p - q).max()) for p, q in zip(fa, fb))
    print(f"[check] gap head, {len(cuts)} cuts: max |dp| {d:.5f}, same top class {same:.3f}; frame head, {len(fa)} "
          f"recordings ({sum(map(len, fa)) / NF.FPS:.0f} s): max |dp| {df:.5f}", flush=True)
    assert d < 5e-3 and same == 1.0 and df < 5e-3, (d, same, df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--heads-dir", type=Path, default=Path(ser.NV_HEADS_DIR))
    ap.add_argument("--frames-dir", type=Path, default=Path(NF.NV_FRAMES_DIR))
    ap.add_argument("--check", type=Path, default=None, help="speech to compare the files and the release on")
    ap.add_argument("--n", type=int, default=64, help="cuts the gap head is compared on")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(REPO, revision=args.revision, allow_patterns=["*.json", "*.py", "model.safetensors"]))
    convert(snap, args.revision, args.heads_dir, args.frames_dir)
    if args.check:
        check(snap, args.heads_dir, args.frames_dir, args.check, args.n)
    print("FROM_RELEASE_DONE", flush=True)


if __name__ == "__main__":
    main()
