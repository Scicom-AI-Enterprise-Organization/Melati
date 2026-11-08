#!/usr/bin/env python3
"""Second follow-up of the release audit, against the run's own files: which language the ASR model detected before a
row was decoded in the language it was released in (melati.pipeline.transcribe falls back to the allowed language
with the best average log-probability), what the fill stage's "dropped words" are, and what lies behind a row with
more characters a second than anyone speaks. RUNS ON THE BOX.

    python scripts/release_language_fallback.py --hf $MELATI_DATA/podcast/hf --work $MELATI_DATA/podcast/work
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    cols = ["id", "recording", "start", "end", "duration", "language", "text", "asr_text", "align_score",
            "asr_avg_logprob", "n_dropped_words", "annotation"]
    rows = [r for f in sorted((a.hf / "data").glob("train-*.parquet")) for r in pq.read_table(f, columns=cols).to_pylist()]
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["recording"]].append(r)

    cross, eg, gap_words, gap_kinds, n_gap_word_rows = Counter(), defaultdict(list), Counter(), Counter(), 0
    fast = []
    for rec, rs in by_rec.items():
        asr = {x["seg"]: x for x in map(json.loads, (a.work / rec / "asr.jsonl").read_text().splitlines())}
        fill = {x["seg"]: x for x in map(json.loads, (a.work / rec / "fill.jsonl").read_text().splitlines())}
        segs = {x["seg"]: x for x in map(json.loads, (a.work / rec / "segments.jsonl").read_text().splitlines())}
        for r in rs:
            det = asr[r["id"]].get("detected_language")
            key = (r["language"], det if det in ("ms", "en", "zh", "ta") else f"other:{det}")
            cross[key] += 1
            if len(eg[key]) < 4 and random.Random(len(eg[key]) + hash(r["id"]) % 997).random() < 0.05:
                eg[key].append((r["id"], r["text"][:70]))
            words = [g for g in fill.get(r["id"], {}).get("gaps") or [] if g.get("kind") == "word"]
            n_gap_word_rows += bool(words)
            for g in fill.get(r["id"], {}).get("gaps") or []:
                gap_kinds[g.get("kind")] += 1
            for g in words:
                gap_words[g["text"].strip().lower()] += 1
            cps = len(r["text"].replace(" ", "")) / max(r["duration"], 0.01)
            if cps > 25 and len(fast) < 400:
                s = segs[r["id"]]
                ws = json.loads(r["annotation"]).get("words") or []
                fast.append((r, s, ws, cps))
    print("language released x language detected first (rows):")
    for (lang, det), k in sorted(cross.items(), key=lambda kv: (-sum(v for (l2, _), v in cross.items() if l2 == kv[0][0]), -kv[1])):
        print(f"  {lang} <- {det:12s} {k:7d}   {eg[(lang, det)][:2]}")
    print(f"\ngaps by kind {dict(gap_kinds)}; rows with a 'word' gap {n_gap_word_rows}")
    print("most common 'word' gap texts:", gap_words.most_common(40))
    total = sum(gap_words.values())
    one = sum(k for t, k in gap_words.items() if len(t.split()) == 1)
    print(f"'word' gaps: {total}; one word {one}; distinct texts {len(gap_words)}")

    print(f"\nrows over 25 chars/s (first {len(fast)} looked at):")
    seg_len = [s["end"] - s["start"] for _, s, _, _ in fast]
    span = [(ws[-1]["end"] - ws[0]["start"]) if ws else 0 for _, _, ws, _ in fast]
    print(f"  their segment before the cut, s: median {sorted(seg_len)[len(seg_len) // 2]:.2f};"
          f" aligned word span median {sorted(span)[len(span) // 2]:.2f}")
    print(f"  segment under 1 s: {sum(x < 1 for x in seg_len)}, 1-2 s: {sum(1 <= x < 2 for x in seg_len)},"
          f" over 2 s: {sum(x >= 2 for x in seg_len)}")
    for r, s, ws, cps in random.Random(0).sample(fast, 12):
        print(f"  {r['id']} {cps:.0f} c/s: segment {s['end'] - s['start']:.2f} s (islands {s.get('islands')}),"
              f" released {r['duration']:.2f} s, words {[(w['text'], w['start'], w['end']) for w in ws][:8]}"
              f"\n      text {r['text'][:100]!r}")


if __name__ == "__main__":
    main()
