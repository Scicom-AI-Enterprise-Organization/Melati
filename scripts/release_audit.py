#!/usr/bin/env python3
"""Is the release good? Every published shard's columns (not the audio) checked for what would make a row wrong
rather than merely noisy: rows that repeat or overlap, text that cannot be what the audio says (empty, a loop, a
rate no one speaks at, a script the language does not use, a stock phrase repeated across recordings), cuts and
alignment, tags, levels and the speaker ids. scripts/release_audio_check.py looks at the audio itself. RUNS ON THE BOX.

    python scripts/release_audit.py --hf $MELATI_DATA/podcast/hf --out $MELATI_DATA/results/dataset/release_audit.json
"""
import argparse
import json
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TAG = re.compile(r"<(/?)([a-z]+)>")
POINT = re.compile(r"\[([a-z]+)\]")
SCRIPTS = ("LATIN", "CJK", "TAMIL", "ARABIC", "THAI", "DEVANAGARI", "CYRILLIC", "HANGUL", "HIRAGANA", "KATAKANA")
EXPECT = {"ms": "LATIN", "en": "LATIN", "ta": "TAMIL", "zh": "CJK"}
_script = {}


def script_of(ch):
    if ch not in _script:
        name = unicodedata.name(ch, "")
        _script[ch] = next((s for s in SCRIPTS if name.startswith(s)), "OTHER")
    return _script[ch]


def dominant_script(text):
    c = Counter(script_of(ch) for ch in text if ch.isalpha())
    return c.most_common(1)[0][0] if c else "NONE"


def gzip_ratio(t):
    b = t.encode()
    return len(b) / len(zlib.compress(b)) if b else 0.0


def q(v, ps=(1, 10, 50, 90, 99)):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    return {str(p): round(float(np.percentile(v, p)), 3) for p in ps} if len(v) else {}


def share(n, d):
    return round(100.0 * n / max(1, d), 2)


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = sorted((a.hf / "data").glob("train-*.parquet"))
    cols = [c for c in pq.read_schema(files[0]).names if c != "audio"]
    rows = [r for f in files for r in pq.read_table(f, columns=cols).to_pylist()]
    n = len(rows)
    hours = sum(r["duration"] for r in rows) / 3600
    R = {"shards": len(files), "rows": n, "hours": round(hours, 1),
         "recordings": len({r["recording"] for r in rows}), "speakers": len({r["speaker"] for r in rows})}

    # integrity: ids, nulls, times, neighbours
    ids = Counter(r["id"] for r in rows)
    R["duplicate_ids"] = sum(1 for v in ids.values() if v > 1)
    R["nulls"] = {c: k for c in cols if (k := sum(r[c] is None for r in rows))}
    R["end_minus_start_minus_duration"] = q([r["end"] - r["start"] - r["duration"] for r in rows])
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["recording"]].append(r)
    overlaps, overlap_s, bad_links = 0, 0.0, 0
    for rs in by_rec.values():
        rs.sort(key=lambda r: r["start"])
        for x, y in zip(rs, rs[1:]):
            if y["start"] < x["end"] - 0.01:
                overlaps += 1
                overlap_s += x["end"] - y["start"]
            if x["next"] != y["id"] or y["prev"] != x["id"]:
                bad_links += 1
    R["overlapping_neighbours"] = {"pairs": overlaps, "seconds": round(overlap_s, 1)}
    R["prev_next_mismatch"] = bad_links

    # durations, by language too
    dur = [r["duration"] for r in rows]
    R["duration_s"] = q(dur)
    R["duration_bands"] = {b: share(sum(lo <= d < hi for d in dur), n) for b, (lo, hi) in
                           {"<1": (0, 1), "1-3": (1, 3), "3-30": (3, 30.0001), ">30": (30.0001, 1e9)}.items()}
    langs = defaultdict(list)
    for r in rows:
        langs[r["language"]].append(r)
    R["by_language"] = {k: {"rows": len(v), "hours": round(sum(r["duration"] for r in v) / 3600, 2),
                            "duration_s": q([r["duration"] for r in v], (10, 50, 90)),
                            "scripts": dict(Counter(dominant_script(r["text"] or "") for r in v).most_common(6))}
                        for k, v in sorted(langs.items(), key=lambda kv: -len(kv[1]))}

    # text that cannot be what the audio says
    empty = [r for r in rows if not (r["text"] or "").strip()]
    cps = [len((r["text"] or "").replace(" ", "")) / max(r["duration"], 0.01) for r in rows]
    slow = [r for r, c in zip(rows, cps) if c < 2 and r["duration"] >= 3 and (r["text"] or "").strip()]
    fast = [r for r, c in zip(rows, cps) if c > 25]
    loops = [r for r in rows if gzip_ratio(r["text"] or "") > 2.4]
    wrong_script = [r for r in rows if (r["text"] or "").strip()
                    and dominant_script(r["text"]) not in (EXPECT.get(r["language"]), "NONE")]
    R["text"] = {
        "empty": len(empty), "empty_pct": share(len(empty), n),
        "empty_hours": round(sum(r["duration"] for r in empty) / 3600, 2),
        "chars_per_s": q(cps), "slow_lt2cps_ge3s": len(slow), "fast_gt25cps": len(fast),
        "gzip_loops": len(loops), "asr_looped": sum(bool(r["asr_looped"]) for r in rows),
        "asr_fallback": sum((r["asr_fallback_t"] or 0) > 0 for r in rows),
        "script_not_of_language": len(wrong_script),
        "script_not_of_language_by": dict(Counter(f"{r['language']}:{dominant_script(r['text'])}"
                                                  for r in wrong_script).most_common(10)),
        "asr_avg_logprob": q([r["asr_avg_logprob"] for r in rows]),
        "asr_avg_logprob_lt_-1": sum((r["asr_avg_logprob"] or 0) < -1 for r in rows),
        "examples": {
            "slow": [(r["id"], r["duration"], r["text"][:120]) for r in slow[:6]],
            "fast": [(r["id"], r["duration"], r["text"][:120]) for r in fast[:6]],
            "loops": [(r["id"], r["duration"], r["text"][:120]) for r in loops[:6]],
            "wrong_script": [(r["id"], r["language"], r["text"][:120]) for r in wrong_script[:10]],
            "empty": [(r["id"], r["duration"], r["language"], r["mos"]) for r in empty[:6]],
        },
    }
    texts = Counter((r["text"] or "").strip() for r in rows if (r["text"] or "").strip())
    R["most_common_texts"] = texts.most_common(25)
    rec_of = defaultdict(set)
    for r in rows:
        t = (r["text"] or "").strip()
        if len(t) >= 60:
            rec_of[t].add(r["recording"])
    dup = {t: s for t, s in rec_of.items() if len(s) > 1}
    R["long_text_in_several_recordings"] = {"texts": len(dup), "rows": sum(texts[t] for t in dup),
                                            "examples": [(t[:100], sorted(s)[:4]) for t, s in list(dup.items())[:8]]}
    yt = defaultdict(set)
    for r in rows:
        if r["youtube_id"]:
            yt[r["youtube_id"]].add(r["recording"])
    R["youtube_id_in_several_recordings"] = {k: sorted(v) for k, v in yt.items() if len(v) > 1}
    R["recordings_without_youtube_id"] = len({r["recording"] for r in rows if not r["youtube_id"]})

    # cuts and alignment
    first, last = [], []
    for r in rows:
        ws = json.loads(r["annotation"]).get("words") or []
        if ws:
            first.append(ws[0]["start"])
            last.append(r["duration"] - ws[-1]["end"])
    R["align"] = {"align_score": q([r["align_score"] for r in rows]),
                  "edge_clip_start_pct": share(sum(bool(r["edge_clip_start"]) for r in rows), n),
                  "edge_clip_end_pct": share(sum(bool(r["edge_clip_end"]) for r in rows), n),
                  "forced_cut_pct": share(sum(bool(r["forced_cut"]) for r in rows), n),
                  "overlap_removed_pct": share(sum((r["overlap_s_removed"] or 0) > 0 for r in rows), n),
                  "first_word_start_s": q(first), "after_last_word_s": q(last),
                  "dropped_words_pct": share(sum((r["n_dropped_words"] or 0) > 0 for r in rows), n)}

    # tags: well formed, and how they are spread
    open_close, unknown_points, emo_rows = Counter(), Counter(), Counter()
    unbalanced = 0
    for r in rows:
        t = r["tagged_text"] or ""
        stack, ok = [], True
        for close, name in TAG.findall(t):
            open_close[name] += not close
            if close:
                ok = ok and bool(stack) and stack.pop() == name
            else:
                stack.append(name)
        unbalanced += not ok or bool(stack)
        for p in POINT.findall(t):
            unknown_points[p] += 1
        for name in set(nm for c, nm in TAG.findall(t) if not c):
            emo_rows[name] += 1
    R["tags"] = {"emotion_spans": dict(open_close), "rows_with_emotion": dict(emo_rows),
                 "rows_with_any_emotion_pct": share(sum(1 for r in rows if TAG.search(r["tagged_text"] or "")), n),
                 "point_tags": dict(unknown_points), "unbalanced_rows": unbalanced,
                 "emotion_segment": dict(Counter(r["emotion_segment"] for r in rows).most_common()),
                 "laughs_per_h": round(sum(r["n_laughs"] for r in rows) / hours, 1),
                 "breaths_per_h": round(sum(r["n_breaths"] for r in rows) / hours, 1),
                 "fillers_per_h": round(sum(r["n_fillers"] for r in rows) / hours, 1)}

    # levels and quality
    R["quality"] = {"mos": q([r["mos"] for r in rows]), "dnsmos": q([r["dnsmos"] for r in rows]),
                    "mos_lt2_pct": share(sum((r["mos"] or 0) < 2 for r in rows), n),
                    "lufs": q([r["lufs"] for r in rows]), "peak_db": q([r["peak_db"] for r in rows]),
                    "peak_ge_-0.1_pct": share(sum((r["peak_db"] if r["peak_db"] is not None else -99) >= -0.1
                                                  for r in rows), n),
                    "lufs_lt_-40_pct": share(sum((r["lufs"] if r["lufs"] is not None else 0) < -40 for r in rows), n),
                    "audio_version": dict(Counter(r["audio_version"] for r in rows))}

    # speakers
    spk = Counter(r["speaker"] for r in rows)
    rec_spk = defaultdict(set)
    for r in rows:
        rec_spk[r["recording"]].add(r["speaker"])
    R["speakers_detail"] = {"segments_per_speaker": q(list(spk.values())),
                            "speakers_with_one_segment_pct": share(sum(v == 1 for v in spk.values()), len(spk)),
                            "speaker_ids_per_recording": q([len(v) for v in rec_spk.values()]),
                            "recordings_with_one_speaker_pct": share(sum(len(v) == 1 for v in rec_spk.values()),
                                                                     len(rec_spk))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1, ensure_ascii=False))
    ex = rows[len(rows) // 2]
    print("\nexample row:", json.dumps({k: (v[:400] if isinstance(v, str) else v) for k, v in ex.items()},
                                       ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
