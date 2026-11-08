"""Re-send transcripts the run kept as empty, exactly as the pipeline sent them, and tally what the ASR server says.

vLLM answers an exception it did not expect with HTTP 200 and an {"error": ...} body, which the client read as no
text; this checks that every empty transcript is the benign case (a decode that is only end-of-text, which vLLM
0.17's verbose_json segment builder fails on with "tuple index out of range") and not a server failure.

    python scripts/probe_empty_asr.py --work $MELATI_DATA/podcast/work --n 400
"""
import argparse
import asyncio
import collections
import io
import json
import random
from pathlib import Path

import aiohttp
import soundfile as sf

from melati.pipeline import SR16, TOKENS_MIN, TOKENS_PER_S


def clip(path, t0, t1):
    """Only the clip's own samples: a whole recording read per clip held 20 GB for 800 of them."""
    x, _ = sf.read(str(path), start=max(0, int(t0 * SR16)), stop=int(t1 * SR16), dtype="float32")
    return x


async def send(session, sem, url, x, cap):
    buf = io.BytesIO()
    sf.write(buf, x, SR16, format="WAV", subtype="PCM_16")
    f = aiohttp.FormData()
    f.add_field("file", buf.getvalue(), filename="seg.wav", content_type="audio/wav")
    f.add_field("model", "melati-asr")
    f.add_field("response_format", "verbose_json")
    f.add_field("temperature", "0.0")
    f.add_field("max_completion_tokens", str(cap))
    async with sem, session.post(f"{url}/v1/audio/transcriptions", data=f) as r:
        body = await r.json(content_type=None)
    err = body.get("error") if isinstance(body, dict) else None
    if err is not None:
        return f"{r.status} error: {(err.get('message') if isinstance(err, dict) else str(err))[:80]}", ""
    text = (body.get("text") or "").strip()
    return (f"{r.status} text" if text else f"{r.status} no text"), text


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--urls", nargs="*", default=["http://127.0.0.1:18100", "http://127.0.0.1:18101",
                                                   "http://127.0.0.1:18102"])
    ap.add_argument("--n", type=int, default=400, help="empty segments, and as many empty gaps")
    ap.add_argument("--pad", type=float, default=0.10)
    a = ap.parse_args()
    recs = sorted(p.parent for p in a.work.glob("*/fill.jsonl"))
    random.Random(0).shuffle(recs)
    jobs = {"segment": [], "gap": []}
    for d in recs:
        if all(len(v) >= a.n for v in jobs.values()):
            break
        segs = {s["seg"]: s for s in map(json.loads, (d / "segments.jsonl").read_text().splitlines())}
        empty = [r["seg"] for r in map(json.loads, (d / "asr.jsonl").read_text().splitlines())
                 if not (r.get("text") or "").strip()][:5]
        gaps = [(row["seg"], g) for row in map(json.loads, (d / "fill.jsonl").read_text().splitlines())
                for g in row.get("gaps") or [] if g.get("kind") == "empty"][:3]
        if len(jobs["segment"]) < a.n:
            for sid in empty:
                s = segs[sid]
                jobs["segment"].append((clip(d / "speech16k.flac", s["start"], s["end"]),
                                        int(TOKENS_MIN + TOKENS_PER_S * (s["end"] - s["start"])), f"{d.name}/{sid}"))
        if len(jobs["gap"]) < a.n:
            for sid, g in gaps:
                t0, t1 = segs[sid]["start"] + g["s"] - a.pad, segs[sid]["start"] + g["e"] + a.pad
                jobs["gap"].append((clip(d / "vocal16k.flac", t0, t1),
                                    int(TOKENS_MIN + TOKENS_PER_S * (t1 - t0)), f"{d.name}/{sid}"))
    sem = asyncio.Semaphore(150)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        for kind, js in jobs.items():
            res = await asyncio.gather(*[send(session, sem, a.urls[i % len(a.urls)], x, cap)
                                         for i, (x, cap, _) in enumerate(js)])
            tally = collections.Counter(r for r, _ in res)
            print(f"empty {kind}s re-sent: {len(js)}")
            for k, n in tally.most_common():
                print(f"  {n:5d}  {k}")
            for (r, text), (_, _, name) in zip(res, js):
                if r.endswith("text") and text:
                    print(f"    now has text: {name}: {text[:80]!r}")


if __name__ == "__main__":
    asyncio.run(main())
