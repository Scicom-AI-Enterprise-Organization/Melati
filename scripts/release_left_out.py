#!/usr/bin/env python3
"""What the release leaves out (melati.stage_export.release_drop) over every finished recording, and why, before a
re-pack: rows and hours by reason and language, the Tamil and Mandarin rows it keeps, and the squeezed-word counts.
A few examples of each are printed, never written: they are transcripts. RUNS ON THE BOX.

    python scripts/release_left_out.py --sources $MELATI_DATA/podcast/sources.jsonl \\
        --work $MELATI_DATA/podcast/work --out $MELATI_DATA/results/dataset/left_out.json
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WORK = None


def one(rec):
    from melati.pipeline import export_rows
    from melati.stage_export import DEFAULTS, build, load_jsonl, release_drop
    d = WORK / rec["id"]
    dropped = {}
    kept = export_rows(WORK, rec, {}, audio=False, dropped=dropped)
    # the left-out rows again, for their language and an example (export_rows only counts them)
    asr = load_jsonl(d / "asr.jsonl")
    said = [r for r in asr.values() if r.get("text")]
    heard = {k: v / len(said) for k, v in Counter(r.get("detected_language") for r in said).items()} if said else {}
    out = []
    if dropped:
        import numpy as np
        segs = [json.loads(line) for line in (d / "segments.jsonl").open()]
        al, mos = load_jsonl(d / "align.jsonl"), load_jsonl(d / "mos.jsonl")
        tag, fill = load_jsonl(d / "tags.jsonl"), load_jsonl(d / "fill.jsonl")
        stems = dict(np.load(d / "stems.npz"))
        for s in segs:
            a = asr.get(s["seg"])
            if not (a or {}).get("text"):
                continue
            row, _, _ = build(s, a, al.get(s["seg"]), mos.get(s["seg"]), tag.get(s["seg"]), fill.get(s["seg"]),
                              stems, dict(DEFAULTS))
            why = release_drop(row, a, heard, DEFAULTS)
            if why:
                out.append((why, row["language"], a.get("detected_language"), row["duration"], row["text"][:90]))
    keep = [(r["language"], (asr.get(r["id"]) or {}).get("detected_language"), r["duration"], r["squeezed_words"],
             r["text"][:90]) for r, _ in kept]
    return rec["id"], out, keep


def main():
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    WORK = a.work
    recs = [json.loads(line) for line in a.sources.open()]
    recs = [r for r in recs if (a.work / r["id"] / "fill.jsonl").exists()]
    by_why, by_lang, eg = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0]), defaultdict(list)
    kept_lang, kept_eg, squeezed = defaultdict(lambda: [0, 0.0]), defaultdict(list), Counter()
    kept_heard, heard_eg = defaultdict(lambda: [0, 0.0]), defaultdict(list)
    rnd = random.Random(0)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.work,)) as ex:
        for k, (rid, out, keep) in enumerate(ex.map(one, recs, chunksize=8), 1):
            for why, lang, det, dur, text in out:
                by_why[why][0] += 1
                by_why[why][1] += dur
                by_lang[f"{why} {lang} (heard {det})"][0] += 1
                by_lang[f"{why} {lang} (heard {det})"][1] += dur
                if len(eg[why]) < 12 or rnd.random() < 0.01:
                    eg[why].append((lang, det, round(dur, 1), text))
            for lang, det, dur, sq, text in keep:
                kept_lang[lang][0] += 1
                kept_lang[lang][1] += dur
                k2 = f"{lang} (heard {det if det in ('ms', 'en', 'id', None) else 'other'})"
                kept_heard[k2][0] += 1
                kept_heard[k2][1] += dur
                if det not in ("ms", "en") and (len(heard_eg[k2]) < 12 or rnd.random() < 0.002):
                    heard_eg[k2].append((round(dur, 1), text))
                squeezed[min(sq, 3)] += 1
                if lang in ("ta", "zh") and len(kept_eg[lang]) < 15:
                    kept_eg[lang].append((rid, round(dur, 1), text))
            if k % 1000 == 0:
                print(f"{k}/{len(recs)} recordings", flush=True)
    hours = lambda v: {"rows": v[0], "hours": round(v[1] / 3600, 2)}  # noqa: E731
    res = {"recordings": len(recs), "left_out": {k: hours(v) for k, v in sorted(by_why.items())},
           "left_out_by_language": {k: hours(v) for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1][0])},
           "kept_by_language": {k: hours(v) for k, v in sorted(kept_lang.items(), key=lambda kv: -kv[1][0])},
           "kept_by_language_heard": {k: hours(v) for k, v in sorted(kept_heard.items(), key=lambda kv: -kv[1][0])},
           "squeezed_words": {("3+" if k == 3 else str(k)): v for k, v in sorted(squeezed.items())}}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    for why, xs in eg.items():
        print(f"\n== {why}: examples")
        for x in rnd.sample(xs, min(10, len(xs))):
            print("  ", x)
    for k2, xs in heard_eg.items():
        print(f"\n== kept {k2}: examples")
        for x in rnd.sample(xs, min(12, len(xs))):
            print("  ", x)
    for lang, xs in kept_eg.items():
        print(f"\n== kept {lang}: examples")
        for x in xs:
            print("  ", x)


def _init(work):
    global WORK
    WORK = work


if __name__ == "__main__":
    main()
