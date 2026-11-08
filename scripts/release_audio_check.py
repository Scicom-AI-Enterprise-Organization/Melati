#!/usr/bin/env python3
"""The released audio itself, on a random sample of rows read back from the published shards: that it decodes at
24 kHz mono to the length its row says, how loud its first and last 20 ms are against its speech (a cut inside a
word starts loud), that Melati's own ASR model given the released clip writes what the row's `asr_text` says (the
pairing of audio and text, and the cut), and how far stock whisper-large-v3, an independent model, is from `text`.
RUNS ON THE BOX (GPU 1 for whisper, the idle melati-asr servers for the rest).

    CUDA_VISIBLE_DEVICES=1 HF_HOME=$MELATI_DATA/hf HF_HUB_OFFLINE=1 python scripts/release_audio_check.py \\
        --hf $MELATI_DATA/podcast/hf --groups 300 --out $MELATI_DATA/results/dataset/release_audio_check.json
"""
import argparse
import asyncio
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def norm(t):
    from melati.tags import is_filler
    t = PUNCT.sub(" ", (t or "").lower())
    return [w for w in t.split() if not is_filler(w)]


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def rates(ref, hyp):
    r, h = norm(ref), norm(hyp)
    rc, hc = "".join(r), "".join(h)
    wer = edits(r, h) / max(1, len(r))
    cer = edits(list(rc), list(hc)) / max(1, len(rc))
    return round(wer, 3), round(cer, 3)


def edge_db(x):
    f = int(0.02 * SR)
    frames = np.array([np.sqrt(np.mean(x[i:i + f] ** 2) + 1e-12) for i in range(0, len(x) - f + 1, f)])
    if len(frames) < 3:
        return None, None
    speech = np.percentile(frames, 95)
    return round(float(20 * np.log10(frames[0] / speech)), 1), round(float(20 * np.log10(frames[-1] / speech)), 1)


async def melati_asr(rows, urls):
    import aiohttp
    from scipy.signal import resample_poly
    sem = asyncio.Semaphore(96)

    async def one(i, r):
        buf = io.BytesIO()
        sf.write(buf, resample_poly(r["x"], 2, 3).astype(np.float32), 16000, format="WAV", subtype="PCM_16")
        f = aiohttp.FormData()
        f.add_field("file", buf.getvalue(), filename="seg.wav", content_type="audio/wav")
        f.add_field("model", "melati-asr")
        f.add_field("response_format", "json")
        f.add_field("temperature", "0.0")
        f.add_field("language", r["language"])
        async with sem, session.post(f"{urls[i % len(urls)]}/v1/audio/transcriptions", data=f) as resp:
            body = await resp.json(content_type=None)
        return (body.get("text") or "").strip() if isinstance(body, dict) else ""

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        return await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])


def stock_whisper(rows):
    import torch
    from scipy.signal import resample_poly
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
                   device="cuda:0")
    out = [None] * len(rows)
    for lang in sorted({r["language"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["language"] == lang]
        xs = [{"raw": resample_poly(rows[i]["x"], 2, 3).astype(np.float32), "sampling_rate": 16000} for i in idx]
        res = asr(xs, batch_size=16, generate_kwargs={"language": lang, "task": "transcribe"})
        for i, r in zip(idx, res):
            out[i] = r["text"].strip()
    return out


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=300, help="random row groups; two random rows from each")
    ap.add_argument("--urls", nargs="*", default=["http://127.0.0.1:18100", "http://127.0.0.1:18101",
                                                   "http://127.0.0.1:18102"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rnd = random.Random(0)
    files = sorted((a.hf / "data").glob("train-*.parquet"))
    groups = [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
    cols = ["audio", "id", "duration", "language", "text", "asr_text", "edge_clip_start", "edge_clip_end", "mos"]
    rows = []
    for f, g in rnd.sample(groups, a.groups):
        t = pq.ParquetFile(f).read_row_group(g, columns=cols).to_pylist()
        for r in rnd.sample(t, min(2, len(t))):
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            r.update(x=x, sr=sr, channels=1 if x.ndim == 1 else x.shape[1])
            del r["audio"]
            rows.append(r)
    bad_sr = sum(r["sr"] != SR or r["channels"] != 1 for r in rows)
    len_off = [abs(len(r["x"]) / r["sr"] - r["duration"]) for r in rows]
    edges = [edge_db(r["x"]) for r in rows]
    starts = [e[0] for e in edges if e[0] is not None]
    ends = [e[1] for e in edges if e[1] is not None]
    print(f"rows {len(rows)} from {a.groups} row groups; not 24 kHz mono: {bad_sr};"
          f" length off its duration by > 20 ms: {sum(o > 0.02 for o in len_off)}")
    print(f"first 20 ms within 10 dB of the clip's speech level: {sum(s > -10 for s in starts)} of {len(starts)};"
          f" last 20 ms: {sum(e > -10 for e in ends)}")
    flagged = [r for r, e in zip(rows, edges) if e[0] is not None and e[0] > -10]
    print(f"  of the loud starts, edge_clip_start set: {sum(bool(r['edge_clip_start']) for r in flagged)}")

    mine = asyncio.run(melati_asr(rows, a.urls))
    pair = [rates(r["asr_text"], h) for r, h in zip(rows, mine)]
    pc = np.array([c for _, c in pair])
    print(f"melati-asr on the released clip against asr_text: CER median {np.median(pc):.3f},"
          f" > 0.3: {(pc > 0.3).sum()}, > 0.5: {(pc > 0.5).sum()} of {len(pc)}")
    stock = stock_whisper(rows)
    ind = [rates(r["text"], h) for r, h in zip(rows, stock)]
    iw, ic = np.array([w for w, _ in ind]), np.array([c for _, c in ind])
    ref_words = np.array([len(norm(r["text"])) for r in rows])
    pooled = float((iw * ref_words).sum() / max(1, ref_words.sum()))
    print(f"stock whisper-large-v3 against text: pooled WER {pooled:.3f}; per row WER median {np.median(iw):.3f},"
          f" CER median {np.median(ic):.3f}; CER > 0.5: {(ic > 0.5).sum()} of {len(ic)}")
    for lang in sorted({r["language"] for r in rows}):
        m = np.array([r["language"] == lang for r in rows])
        print(f"  {lang}: {m.sum()} rows, CER median {np.median(ic[m]):.3f}, > 0.5: {(ic[m] > 0.5).sum()}")
    order = np.argsort(-ic)
    print("largest disagreements (text | stock whisper | melati-asr on the clip):")
    for i in order[:25]:
        r = rows[i]
        print(f"  {r['id']} {r['language']} {r['duration']:.1f}s CER {ic[i]:.2f}\n     text  {r['text'][:110]!r}"
              f"\n     stock {stock[i][:110]!r}\n     mine  {mine[i][:110]!r}")
    print("pairing worst (asr_text | melati-asr on the released clip):")
    for i in np.argsort(-pc)[:10]:
        r = rows[i]
        print(f"  {r['id']} {r['duration']:.1f}s CER {pc[i]:.2f}: {r['asr_text'][:90]!r} | {mine[i][:90]!r}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "rows": len(rows), "not_24k_mono": bad_sr, "length_off_20ms": int(sum(o > 0.02 for o in len_off)),
        "loud_start": int(sum(s > -10 for s in starts)), "loud_end": int(sum(e > -10 for e in ends)),
        "pairing_cer_median": float(np.median(pc)), "pairing_cer_gt_0.3": int((pc > 0.3).sum()),
        "stock_pooled_wer": round(pooled, 3), "stock_cer_median": float(np.median(ic)),
        "stock_cer_gt_0.5": int((ic > 0.5).sum()),
        "examples": [{"id": rows[i]["id"], "language": rows[i]["language"], "text": rows[i]["text"],
                      "stock": stock[i], "mine": mine[i], "cer": float(ic[i])} for i in order[:60]]},
        indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
