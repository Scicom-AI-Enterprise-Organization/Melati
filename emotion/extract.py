#!/usr/bin/env python3
"""Frozen features for the emotion probes: every layer of an encoder, mean-pooled over the clip. RUNS ON THE BOX.

One array per encoder and corpus, <out>/<encoder>/<corpus>.npy, shape (clips, layers, dim) in float16, with the
clip ids beside it (<corpus>.ids.json), so a probe that learns its own mix of layers (emotion/probe.py) trains in
seconds, every encoder is compared on the same clips, and a corpus downloaded later is added without redoing the
others (a corpus whose file exists is skipped). Pooling covers the clip's own frames only: Whisper pads every input to 30 s, and averaging the padding
in would make a two-second clip look like silence.

  whisper     openai/whisper-large-v3 encoder (EmoBox's strongest SER backbone)
  whisper_ft  the Melati fine-tune's encoder (--whisper-ft <dir>): the same model after Malaysian podcasts
  wavlm       microsoft/wavlm-large (EmoBox's second)
  w2vbert     facebook/w2v-bert-2.0 (multilingual pre-training that includes Malay)
  e2v         emotion2vec+ large, utterance embedding (one layer; the reference the others must beat)
  ast         AST fine-tuned on AudioSet, pooled over the clip's own patches (the non-verbal heads)

Any manifest with id / corpus / src rows works: nonverbal/build_manifest.py's clips go through it too.

    python emotion/extract.py --manifest $MELATI_DATA/emo/manifest.jsonl --encoder whisper --out $MELATI_DATA/emo/feats
"""
import argparse, io, json, subprocess, time
from pathlib import Path

import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000
MAX_S = 30.0


def ffmpeg(arg, raw=None):
    """16 kHz mono float32 via ffmpeg, from a file path (an mp4 needs a seekable input) or from bytes."""
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0" if raw is not None else str(arg), "-vn", "-f", "f32le",
                        "-ac", "1", "-ar", str(SR), "pipe:1"], input=raw, stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def decode(raw: bytes, path=None):
    if path is not None and Path(path).suffix.lower() in (".mp4", ".m4a", ".mp3", ".webm", ".ogg"):
        return ffmpeg(path)
    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:                              # anything soundfile cannot read: ffmpeg
        return ffmpeg(path, None) if path is not None else ffmpeg(None, raw)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR)
    return x.astype(np.float32)


class Audio:
    """Reads a manifest row's audio: a file on disk, or a cell of a parquet file (the last few files' columns kept: a
    manifest over all of Melati's shards would otherwise hold every shard's audio in memory, about 0.5 GB each)."""
    KEEP = 4

    def __init__(self):
        self.tables = {}

    def __call__(self, src):
        """The clip's samples, or None when it cannot be decoded (MELD ships a few broken videos)."""
        try:
            return self._read(src)
        except Exception as e:  # noqa: BLE001
            print(f"  [undecodable] {src}: {type(e).__name__}", flush=True)
            return None

    def _read(self, src):
        if "file" in src:
            f = Path(src["file"])
            return decode(None if f.suffix.lower() in (".mp4", ".m4a", ".mp3", ".webm", ".ogg") else f.read_bytes(), f)
        import pyarrow.parquet as pq
        t = self.tables.get(src["parquet"])
        if t is None:
            if len(self.tables) >= self.KEEP:
                self.tables.pop(next(iter(self.tables)))
            t = self.tables[src["parquet"]] = pq.read_table(src["parquet"], columns=[src["col"]]).column(0)
        cell = t[src["row"]].as_py()
        return decode(cell["bytes"] if isinstance(cell, dict) else cell)


from melati.ser import ENCODERS  # noqa: E402   the same encoders the tag server runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    ap.add_argument("--whisper-ft", default=None, help="default melati.ser.WHISPER_FT, the encoder the head saw")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8")]
    if args.encoder == "whisper_ft" and args.whisper_ft:
        import melati.ser as S
        S.WHISPER_FT, S.WHISPER_FT_REVISION = args.whisper_ft, None
    run = ENCODERS[args.encoder]("cuda")
    audio = Audio()
    out = args.out / args.encoder
    out.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in rows:
        by.setdefault(r["corpus"], []).append(r)
    for corpus, rs in sorted(by.items()):
        dst = out / f"{corpus}.npy"
        if dst.exists():
            continue
        feats, t0, bad = None, time.time(), []
        for s in range(0, len(rs), args.batch):
            batch = []
            for r in rs[s:s + args.batch]:
                x = audio(r["src"])
                if x is None or len(x) == 0:          # kept as a row so ids stay aligned; probe.py drops it
                    bad.append(r["id"]); x = np.zeros(SR // 10, np.float32)
                batch.append(x[: int(MAX_S * SR)])
            batch = [x if len(x) >= SR // 10 else np.pad(x, (0, SR // 10 - len(x))) for x in batch]
            f = run(batch)
            if feats is None:
                feats = np.lib.format.open_memmap(str(dst) + ".part.npy", mode="w+", dtype=np.float16,
                                                  shape=(len(rs),) + f.shape[1:])
            feats[s:s + len(batch)] = f.astype(np.float16)
        feats.flush()
        del feats
        Path(str(dst) + ".part.npy").rename(dst)
        (out / f"{corpus}.ids.json").write_text(json.dumps([r["id"] for r in rs]))
        if bad:
            (out / f"{corpus}.bad.json").write_text(json.dumps(bad))
        el = time.time() - t0
        print(f"[{args.encoder}] {corpus}: {len(rs)} clips in {el:.0f}s ({len(rs) / max(el, 1e-9):.1f}/s)", flush=True)
    print(f"EXTRACT_DONE {args.encoder}", flush=True)


if __name__ == "__main__":
    main()
