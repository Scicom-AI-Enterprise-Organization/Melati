#!/usr/bin/env python3
"""Does each released clip carry its own transcript? Every row of the given recordings is read back from the published
shards, its clip transcribed again by Melati's ASR model (the idle melati-asr servers), and the result matched against
the `asr_text` of the five rows either side: a row whose clip matches another row's text better than its own is paired
wrongly, and a run of them is an offset between the audio the ASR heard and the audio that was exported. RUNS ON THE BOX.

    python scripts/release_pairing.py --hf $MELATI_DATA/podcast/hf --recordings r08b773934c5a
    python scripts/release_pairing.py --hf $MELATI_DATA/podcast/hf --random 40 --per 12
"""
import argparse
import asyncio
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_audio_check import melati_asr, rates  # noqa: E402


def main():
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--recordings", nargs="*", default=[])
    ap.add_argument("--random", type=int, default=0, help="this many random recordings as well")
    ap.add_argument("--per", type=int, default=0, help="at most this many rows of each (0: all)")
    ap.add_argument("--urls", nargs="*", default=["http://127.0.0.1:18100", "http://127.0.0.1:18101",
                                                   "http://127.0.0.1:18102"])
    a = ap.parse_args()
    files = sorted((a.hf / "data").glob("train-*.parquet"))
    where = defaultdict(list)                    # recording -> [(file, row group)]
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for rec in set(pf.read_row_group(g, columns=["recording"]).column(0).to_pylist()):
                where[rec].append((f, g))
    want = list(a.recordings) + random.Random(0).sample(sorted(set(where) - set(a.recordings)), a.random)
    cols = ["audio", "id", "recording", "start", "duration", "language", "asr_text"]
    shifted_recs, total_rows, total_wrong = [], 0, 0
    for rec in want:
        rows = [r for f, g in where[rec] for r in pq.ParquetFile(f).read_row_group(g, columns=cols).to_pylist()
                if r["recording"] == rec]
        rows.sort(key=lambda r: r["start"])
        if a.per:
            rows = rows[:a.per]
        for r in rows:
            r["x"], _ = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        heard = asyncio.run(melati_asr(rows, a.urls))
        own = np.array([rates(r["asr_text"], h)[1] for r, h in zip(rows, heard)])
        wrong, best_off = 0, []
        for i, h in enumerate(heard):
            if len(h) < 8:
                continue
            near = range(max(0, i - 5), min(len(rows), i + 6))     # a shifted pairing is off by a few rows
            c = {j: rates(rows[j]["asr_text"], h)[1] for j in near}
            j = min(c, key=c.get)
            if j != i and c[j] + 0.2 < own[i]:
                wrong += 1
                best_off.append(j - i)
        total_rows += len(rows)
        total_wrong += wrong
        print(f"{rec}: {len(rows)} rows, own CER median {np.median(own):.3f}, matched another row better: {wrong}"
              + (f" (offsets {sorted(set(best_off))[:8]})" if best_off else ""), flush=True)
        if wrong > max(1, 0.2 * len(rows)):
            shifted_recs.append(rec)
            for i in range(min(4, len(rows))):
                print(f"    {rows[i]['id']}: text {rows[i]['asr_text'][:70]!r}\n        heard {heard[i][:70]!r}")
    print(f"\n{len(want)} recordings, {total_rows} rows; rows matching another row's text better: {total_wrong};"
          f" recordings mostly so: {shifted_recs}")


if __name__ == "__main__":
    main()
