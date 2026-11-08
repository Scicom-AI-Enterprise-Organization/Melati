#!/usr/bin/env python3
"""Rows whose transcript holds words the audio does not: the aligner still has to place such a word and squeezes it
into a few milliseconds (no syllable is shorter than about 50 ms), typically where the ASR model finished a sentence
the cut had stopped. Counted over every published shard, by where in the segment the squeezed words sit. RUNS ON THE BOX.

    python scripts/release_squeezed.py --hf $MELATI_DATA/podcast/hf
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

SQUEEZED_S = 0.025      # a word of three letters or more aligned shorter than this is not in the audio


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    a = ap.parse_args()
    cols = ["id", "duration", "language", "text", "annotation", "asr_avg_logprob", "align_score"]
    rows = [r for f in sorted((a.hf / "data").glob("train-*.parquet")) for r in pq.read_table(f, columns=cols).to_pylist()]
    n, flagged, where, by_lang = len(rows), [], Counter(), Counter()
    lang_n = Counter(r["language"] for r in rows)
    for r in rows:
        ws = [w for w in json.loads(r["annotation"]).get("words") or [] if not w["filler"]]
        sq = [i for i, w in enumerate(ws) if len(w["text"].strip(".,?!")) >= 3 and w["end"] - w["start"] < SQUEEZED_S]
        if not sq:
            continue
        flagged.append((r, len(sq), len(ws)))
        by_lang[r["language"]] += 1
        k = len(ws)
        where["end" if sq[-1] >= k - 2 else "start" if sq[0] <= 1 else "middle"] += 1
    k = np.array([f[1] for f in flagged])
    print(f"rows {n}; with a squeezed word: {len(flagged)} ({100 * len(flagged) / n:.2f}%),"
          f" with 2 or more: {(k >= 2).sum()} ({100 * (k >= 2).sum() / n:.2f}%)")
    print(f"  hours of those rows: {sum(f[0]['duration'] for f in flagged) / 3600:.1f}; where: {dict(where)}")
    print("  share of each language's rows:", {l: f"{100 * by_lang[l] / lang_n[l]:.1f}%" for l in lang_n})
    al = np.array([f[0]["align_score"] or 0 for f in flagged])
    print(f"  their align_score: median {np.median(al):.3f}; under 0.8: {(al < 0.8).sum()}")
    for r, m, k_ in random.Random(0).sample(flagged, 12):
        ws = [w for w in json.loads(r["annotation"]).get("words") or [] if not w["filler"]]
        tail = [(w["text"], round(w["end"] - w["start"], 3)) for w in ws][-6:]
        print(f"  {r['id']} {r['duration']:.2f} s, {m}/{k_} squeezed: {r['text'][:90]!r}\n      last words {tail}")


if __name__ == "__main__":
    main()
