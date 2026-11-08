#!/usr/bin/env python3
"""Follow up scripts/release_audit.py against the run's own files: speech the export's cut left out (the cut follows
the aligner's first and last word, so a bad alignment shortens the audio but not the transcript), where the Tamil and
Mandarin rows come from, how many speakers the diarizer found in recordings released with one speaker id,
recordings that repeat one another, and neighbours whose audio overlaps. RUNS ON THE BOX.

    python scripts/release_cut_and_sources.py --hf $MELATI_DATA/podcast/hf --work $MELATI_DATA/podcast/work
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def q(v, ps=(10, 50, 90, 99)):
    v = np.asarray(v, dtype=float)
    return {str(p): round(float(np.percentile(v, p)), 3) for p in ps} if len(v) else {}


def inside(islands, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in islands)


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    cols = ["id", "recording", "speaker", "start", "end", "duration", "language", "text", "align_score",
            "forced_cut", "overlap_s_removed", "source_title"]
    rows = [r for f in sorted((a.hf / "data").glob("train-*.parquet")) for r in pq.read_table(f, columns=cols).to_pylist()]
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["recording"]].append(r)

    # 1. speech the cut left out: VAD islands of the segment outside the exported span
    lost, cps, low = [], [], []
    for rec, rs in by_rec.items():
        segs = {s["seg"]: s for s in map(json.loads, (a.work / rec / "segments.jsonl").read_text().splitlines())}
        for r in rs:
            s = segs[r["id"]]
            isl = s.get("islands") or [[s["start"], s["end"]]]
            total = inside(isl, s["start"], s["end"])
            kept = inside(isl, r["start"], r["end"])
            lost.append(total - kept)
            cps.append(len(r["text"].replace(" ", "")) / max(r["duration"], 0.01))
            low.append(r["align_score"] if r["align_score"] is not None else 0.0)
    lost, cps, low = map(np.asarray, (lost, cps, low))
    n = len(rows)
    print(f"rows {n}")
    print("speech outside the cut, s:", q(lost))
    for t in (0.3, 0.5, 1.0, 2.0):
        m = lost > t
        print(f"  > {t} s: {m.sum()} rows ({100 * m.mean():.2f}%), {lost[m].sum() / 3600:.2f} h lost;"
              f" of them align_score < 0.8: {(low[m] < 0.8).sum()}, chars/s > 25: {(cps[m] > 25).sum()}")
    fast = cps > 25
    print(f"rows over 25 chars/s: {fast.sum()}; of them with > 0.5 s cut off: {(fast & (lost > 0.5)).sum()};"
          f" align_score median {np.median(low[fast]):.3f} against {np.median(low):.3f} overall")
    worst = np.argsort(-lost)[:6]
    for i in worst:
        r = rows[i]
        print(f"    {r['id']} lost {lost[i]:.1f} s of {r['duration']:.1f}+ s, align {r['align_score']}: {r['text'][:90]!r}")

    # 2. the minority languages: whole recordings, or stray clips inside Malay ones
    for lang in ("ta", "zh", "en"):
        recs = Counter(r["recording"] for r in rows if r["language"] == lang)
        major = {rec for rec, k in recs.items() if k >= 0.5 * len(by_rec[rec])}
        rows_major = sum(recs[rec] for rec in major)
        print(f"{lang}: {sum(recs.values())} rows in {len(recs)} recordings; {len(major)} recordings mostly {lang}"
              f" hold {rows_major} of them; the rest are stray clips in recordings of another language")
        stray = [r for r in rows if r["language"] == lang and r["recording"] not in major]
        major_rows = [r for r in rows if r["language"] == lang and r["recording"] in major]
        rnd = random.Random(0)
        for name, pool in (("stray", stray), ("mostly", major_rows)):
            for r in rnd.sample(pool, min(8, len(pool))):
                print(f"    {name:6s} {r['id']} {r['duration']:.1f}s {r['text'][:80]!r}  [{r['source_title'][:40]}]")

    # 3. speakers: what the diarizer found where the release has one speaker id
    one = [rec for rec, rs in by_rec.items() if len({r["speaker"] for r in rs}) == 1]
    found, durs, modes = Counter(), [], Counter()
    for rec in one:
        d = json.loads((a.work / rec / "diar.json").read_text())
        turns = d.get("turns") or []
        found[min(len({t[2] for t in turns}), 4)] += 1
        modes[d.get("mode")] += 1
        durs.append(json.loads((a.work / rec / "sep.json").read_text())["duration_s"] / 60)
    print(f"recordings released with one speaker id: {len(one)} of {len(by_rec)}; diarizer speakers found"
          f" {dict(sorted(found.items()))} (4 = 4 or more); modes {dict(modes)}; minutes {q(durs)}")
    many = [rec for rec in by_rec if rec not in set(one)]
    dom = []
    for rec in many:
        sp = Counter()
        for r in by_rec[rec]:
            sp[r["speaker"]] += r["duration"]
        dom.append(max(sp.values()) / sum(sp.values()))
    print(f"recordings with several speaker ids: {len(many)}; share of speech by the main speaker {q(dom)}")
    for rec in random.Random(1).sample(one, min(10, len(one))):
        rs = by_rec[rec]
        print(f"    one id: {rec} {len(rs)} segments, {sum(r['duration'] for r in rs) / 60:.1f} min  [{rs[0]['source_title'][:60]}]")

    # 4. recordings that repeat one another: pairs sharing three or more long sentences
    rec_of = defaultdict(set)
    for r in rows:
        t = r["text"].strip()
        if len(t) >= 40:
            rec_of[t].add(r["recording"])
    pairs = Counter()
    for recs in rec_of.values():
        recs = sorted(recs)
        if 1 < len(recs) <= 6:
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    pairs[(recs[i], recs[j])] += 1
    dup = [(p, k) for p, k in pairs.items() if k >= 3]
    hours = lambda rec: sum(r["duration"] for r in by_rec[rec]) / 3600  # noqa: E731
    print(f"recording pairs sharing 3+ long sentences: {len(dup)}; the smaller of each pair holds"
          f" {sum(min(hours(x), hours(y)) for (x, y), _ in dup):.1f} h")
    for (x, y), k in sorted(dup, key=lambda pk: -pk[1])[:10]:
        print(f"    {k:4d} shared: {x} ({hours(x) * 60:.0f} min) [{by_rec[x][0]['source_title'][:35]}] ~"
              f" {y} ({hours(y) * 60:.0f} min) [{by_rec[y][0]['source_title'][:35]}]")

    # 5. neighbours whose audio overlaps: the 0.15 s pads, or speech of the one before
    over, other, deep = 0, 0, 0
    for rs in by_rec.values():
        rs = sorted(rs, key=lambda r: r["start"])
        for x, y in zip(rs, rs[1:]):
            o = x["end"] - y["start"]
            if o > 0.01:
                over += 1
                deep += o > 0.3
                other += o > 0.15 and x["speaker"] != y["speaker"]
    print(f"overlapping neighbours: {over}; by more than both pads (0.3 s): {deep};"
          f" by more than one pad with a different speaker: {other}")


if __name__ == "__main__":
    main()
