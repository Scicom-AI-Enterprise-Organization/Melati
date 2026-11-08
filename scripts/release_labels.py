#!/usr/bin/env python3
"""Two label checks over every published shard that need no audio: how often each emotion's span holds a laugh (an
emotion confused with laughing speech holds them as often as `happy` does), and how often a row's `language` disagrees
with its own words (Malay and English function words counted in the text). RUNS ON THE BOX.

    python scripts/release_labels.py --hf $MELATI_DATA/podcast/hf
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SPAN = re.compile(r"<([a-z]+)>(.*?)</\1>", re.S)
MS = set("yang dan ini itu tak tidak saya aku kita kami dia nak ke di dengan untuk ada lah kan pun macam apa sebab "
         "kalau boleh dah je tu ni orang dalam bila jadi sangat juga tapi atau mana".split())
EN = set("the and is are i you it that this to of in for was what so but we they have do not my be like just "
         "with on can there because think know really".split())


def words(t):
    return re.findall(r"[a-z']+", (t or "").lower())


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    a = ap.parse_args()
    cols = ["id", "recording", "speaker", "language", "text", "tagged_text", "n_laughs", "duration"]
    rows = [r for f in sorted((a.hf / "data").glob("train-*.parquet")) for r in pq.read_table(f, columns=cols).to_pylist()]
    ids = defaultdict(set)
    for r in rows:
        ids[r["recording"]].add(r["speaker"])
    kind = lambda r: "windows" if "_w" in r["speaker"] else "one id" if len(ids[r["recording"]]) == 1 else "several ids"  # noqa: E731
    h = Counter()
    for r in rows:
        h[kind(r)] += r["duration"] / 3600
    print("hours by the recording's speaker ids:", {k: round(v, 1) for k, v in h.items()})
    spans, with_laugh, eg = Counter(), Counter(), defaultdict(list)
    for r in rows:
        for lab, body in SPAN.findall(r["tagged_text"] or ""):
            spans[lab] += 1
            with_laugh[lab] += "[laughs]" in body
            if len(eg[lab]) < 400:
                eg[lab].append(body)
    print("emotion spans holding a laugh:", {k: f"{with_laugh[k]}/{v} ({100 * with_laugh[k] / v:.1f}%)" for k, v in spans.items()})
    base = sum(r["n_laughs"] > 0 for r in rows) / len(rows)
    print(f"rows with any laugh: {100 * base:.1f}%")
    for lab in ("angry", "sad", "fearful"):
        print(f"  {lab}:")
        for body in random.Random(0).sample(eg[lab], min(8, len(eg[lab]))):
            print(f"     {re.sub(r'<[^>]+>', '', body)[:100]!r}")
    mis = Counter()
    sample = defaultdict(list)
    n = Counter(r["language"] for r in rows)
    for r in rows:
        w = words(r["text"])
        m, e = sum(x in MS for x in w), sum(x in EN for x in w)
        if r["language"] == "en" and m >= 2 and m > e:
            mis["en text mostly Malay"] += 1
            sample["en"].append(r["text"])
        if r["language"] == "ms" and e >= 3 and m == 0:
            mis["ms text all English"] += 1
            sample["ms"].append(r["text"])
    print({k: f"{v} ({100 * v / n[k.split()[0]]:.1f}% of {k.split()[0]})" for k, v in mis.items()})
    for k, v in sample.items():
        for t in random.Random(1).sample(v, min(5, len(v))):
            print(f"   {k}: {t[:100]!r}")


if __name__ == "__main__":
    main()
