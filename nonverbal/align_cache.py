#!/usr/bin/env python3
"""NonverbalTTS force-aligned once, so a candidate rule can be changed and re-scored without the aligner. ON THE BOX.

scripts/eval_nonverbal_gaps.py aligned its clips and kept only the candidates; changing the rule that finds them
(melati.nv_gaps.candidates) meant aligning again. This aligns every clip of the splits asked for (the reference
words, stripped as that script strips them) and keeps each clip's word times and reference tags, in row order, so
nonverbal/gap_rule.py can rebuild candidates under any rule for the calibration draw (seed 0, 800 train clips),
the test split, and the training clips outside that draw.

    python nonverbal/align_cache.py --endpoints $MELATI_DATA/serve_align/endpoints.json \\
        --out $MELATI_DATA/nv/nvtts_align.json
"""
import argparse, asyncio, io, json, sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from melati.nv_gaps import SR  # noqa: E402
from scripts.eval_nonverbal import reference  # noqa: E402
from scripts.eval_nonverbal_gaps import strip  # noqa: E402


def rows_of(split):
    """Each row's 16 kHz audio, stripped words and reference tags at word boundaries, as collect() builds them."""
    from datasets import Audio, load_dataset
    d = load_dataset("deepvk/NonverbalTTS", split=split).cast_column("audio", Audio(decode=False))
    for i in range(len(d)):
        r = d[i]
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        raw, tags = reference(r["Result"])
        words = [strip(w) for w in raw]
        kept = [sum(1 for w in words[:k] if w) for k in range(len(words) + 1)]
        yield i, A.resample(x.mean(1), sr, SR), [w for w in words if w], [(cl, kept[sl]) for cl, sl in tags]


async def run(args):
    from melati.client import Client, Endpoints
    out = {}
    async with Client(Endpoints([args.endpoints])) as c:
        for split in args.splits:
            clips, batch = [], []

            async def flush():
                al = await c.align([(x[: 30 * SR], " ".join(ws), "en") for _, x, ws, _ in batch])
                for (i, x, ws, tags), a in zip(batch, al):
                    aw = a.get("words") or []
                    ok = len(aw) == len(ws) and len(x) <= 30 * SR
                    clips.append({"row": i, "dur": round(len(x) / SR, 3), "aligned": ok, "tags": tags,
                                  "words": [{"w": w, "s": round(float(a["s"]), 3), "e": round(float(a["e"]), 3)}
                                            for w, a in zip(ws, aw)] if ok else []})
                batch.clear()
            for item in rows_of(split):
                if item[2]:
                    batch.append(item)
                if len(batch) == 128:
                    await flush()
            if batch:
                await flush()
            out[split] = clips
            print(f"[{split}] {len(clips)} clips, {sum(c['aligned'] for c in clips)} aligned word for word", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"ALIGN_CACHE_DONE -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--out", type=Path, required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
