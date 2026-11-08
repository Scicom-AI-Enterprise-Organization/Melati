#!/usr/bin/env python3
"""Pack Melati's output into the Hugging Face dataset, block by block as the run finishes recordings: parquet shards
of one row per segment with its 24 kHz FLAC inline, every manifest column, and the features metadata that makes
`datasets` decode `audio`. RUNS ON THE BOX.

    python scripts/build_hf_dataset.py --sources $MELATI_DATA/podcast/sources.jsonl \\
        --work $MELATI_DATA/podcast/work --out $MELATI_DATA/podcast/hf --workers 16 [--final]

The source list is cut into fixed blocks of --block recordings, in its own order, and a block becomes one shard,
<out>/data/train-BBBBB.parquet, once every recording in it has finished every stage (with --final, once the run is
over: a recording that failed for good is left out and listed in stats.json). A shard is written once, so a re-run
packs only the blocks finished since, and an upload sends only those. The rows are melati.pipeline.export_rows, what
the export stage writes as files, here encoded in memory. <out>/stats.json (scripts/dataset_card.py turns it into the
card and its figures) is recomputed over every shard on each run; <out>/export_config.json holds the thresholds every
tag was decided at.
"""
import argparse
import io
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 24000
SOURCE = "malaysia-ai/malaysian-podcast-youtube"
EMPTY = 1024        # bytes: the source holds 50 four-byte placeholder files with no audio; no block waits for them
YT = re.compile(r"\s*\[([A-Za-z0-9_-]{11})\]$")
WINDOWED = re.compile(r"_w\d+S\d+$")   # a speaker id of a recording diarized in ten-minute windows (melati/diarize.py)
DROP = ("audio", "asr_no_speech_prob", "source_stem")       # rebuilt below, never filled, or split in two


def features():
    from datasets import Audio, Features, Value
    f = {"audio": Audio(sampling_rate=SR)}
    s, d, i, b = Value("string"), Value("float64"), Value("int64"), Value("bool")
    for k, t in [("id", s), ("recording", s), ("speaker", s), ("start", d), ("end", d), ("duration", d),
                 ("language", s), ("text", s), ("asr_text", s), ("tagged_text", s), ("tagged_text_hesitates", s),
                 ("annotation", s), ("emotion_segment", s), ("emotion_segment_p", d),
                 ("sounds", [{"label": s, "p": d}]), ("n_sound_tags", i), ("n_events", i),
                 ("n_breaths", i), ("n_laughs", i), ("n_coughs", i), ("n_sniffs", i), ("n_throat_clears", i),
                 ("whispered", b), ("n_fillers", i), ("n_dropped_words", i), ("squeezed_words", i),
                 ("audio_version", s), ("background_db", d), ("mos", d), ("mos_vocal", d), ("mos_mix", d),
                 ("dnsmos", d), ("lufs", d), ("peak_db", d), ("align_score", d), ("edge_clip_start", b),
                 ("edge_clip_end", b), ("forced_cut", b), ("overlap_s_removed", d), ("asr_avg_logprob", d),
                 ("asr_fallback_t", d), ("asr_looped", b), ("prev", s), ("next", s), ("source", s),
                 ("source_title", s), ("youtube_id", s)]:
        f[k] = t
    return Features(f)


def row_of(r, flac: bytes):
    ann = json.loads(r["annotation"])
    kinds = Counter(e["kind"] for e in ann.get("events") or [])
    stem = r.get("source_stem") or ""
    m = YT.search(stem)
    out = {k: v for k, v in r.items() if k not in DROP}
    out.update(audio={"bytes": flac, "path": f"{r['recording']}/{r['id']}.flac"}, n_breaths=kinds["breathes"],
               n_laughs=kinds["laughs"], n_coughs=kinds["coughs"], n_sniffs=kinds["sniffs"],
               n_throat_clears=kinds["clears throat"],
               whispered=any(d["label"] == "whisper" for d in ann.get("delivery") or []), source=SOURCE,
               source_title=YT.sub("", stem) if m else stem, youtube_id=m.group(1) if m else None)
    return out


def shard_path(out: Path, b: int) -> Path:
    return out / "data" / f"train-{b:05d}.parquet"


def left_out_path(out: Path, b: int) -> Path:
    """What a shard's block left out of the release, by reason (melati.stage_export.release_drop): kept beside the
    shards, never uploaded, read back into stats.json."""
    return out / "left_out" / f"train-{b:05d}.json"


def pack(job):
    """One block -> one shard: every finished recording's segments, FLAC-encoded in memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from melati.pipeline import export_rows
    b, recs, work, out = job
    rows, dropped = [], {}
    for rec in recs:
        if not (Path(work) / rec["id"] / "fill.jsonl").exists():
            continue
        for r, x in export_rows(Path(work), rec, {}, dropped=dropped):
            buf = io.BytesIO()
            sf.write(buf, np.clip(x, -1.0, 1.0), SR, format="FLAC", subtype="PCM_16")
            rows.append(row_of(r, buf.getvalue()))
    left = left_out_path(Path(out), b)
    left.parent.mkdir(parents=True, exist_ok=True)
    left.write_text(json.dumps(dropped))
    path = shard_path(Path(out), b)
    tmp = path.with_suffix(".part")
    pq.write_table(pa.Table.from_pylist(rows, schema=features().arrow_schema), tmp, compression="zstd",
                   row_group_size=100)
    tmp.replace(path)
    return b, len(rows), path.stat().st_size


def stats(rows, src):
    """What the card says, from every packed shard's columns (not the audio)."""
    h = lambda xs: round(sum(xs) / 3600, 2)  # noqa: E731
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"] or "none"].append(r["duration"])
    ev, emo, dlv, by_src, fill, laugh_over = Counter(), Counter(), Counter(), Counter(), 0, 0
    snd, snd_rows = Counter(), 0                    # AudioSet sound tags, apart from the non-verbal ones
    for r in rows:
        ann = json.loads(r["annotation"])
        for d in ann.get("delivery") or []:
            dlv[d["label"]] += 1
        kinds = [e["kind"] for e in ann.get("events") or [] if e.get("source") == "sounds"]
        snd.update(kinds)
        snd_rows += bool(kinds)
        for e in ann.get("events") or []:
            if e.get("source") == "sounds":
                continue
            ev[e["kind"]] += 1
            by_src[f"{e['kind']}/{e.get('source', '')}"] += 1
            if e["kind"] == "laughs" and e.get("overlap", 0) > 0.5:
                laugh_over += 1
        for s in ann.get("emotion") or []:
            emo[s["label"]] += 1
        fill += sum(1 for w in ann.get("words") or [] if w.get("filler"))
    dur = np.array([r["duration"] for r in rows])
    hours = float(dur.sum() / 3600)
    q = lambda key: [r[key] for r in rows if r.get(key) is not None]  # noqa: E731
    pct = lambda v, qs: {str(p): round(float(np.percentile(v, p)), 3) for p in qs} if len(v) else {}  # noqa: E731
    emilia = [r for r in rows if 3 <= r["duration"] <= 30 and (r["dnsmos"] or 0) >= 3]
    windowed = {r["recording"] for r in rows if WINDOWED.search(r["speaker"])}     # speaker ids per window
    # real rows for the card's example: an emotion scope and a non-verbal tag in one short, well-scored segment
    rich = [r for r in rows if r["tagged_text"].startswith("<") and "[" in r["tagged_text"]
            and len(r["tagged_text"]) <= 140 and (r["mos"] or 0) >= 3 and r["language"] == "ms"]
    rich.sort(key=lambda r: -((r["align_score"] or 0) + (r["mos"] or 0)))
    return dict(src, **{
        "recordings_with_rows": len({r["recording"] for r in rows}), "segments": len(rows),
        "hours": round(hours, 1), "speakers": len({r["speaker"] for r in rows}),
        "hours_by_language": {k: h(v) for k, v in sorted(by_lang.items(), key=lambda kv: -sum(kv[1]))},
        "segments_by_language": {k: len(v) for k, v in sorted(by_lang.items(), key=lambda kv: -len(kv[1]))},
        "duration_s": pct(dur, (1, 10, 50, 90, 99)),
        "duration_hist": np.histogram(dur, bins=np.arange(0, 31, 1))[0].tolist(),
        "events": dict(ev), "events_per_hour": {k: round(v / hours, 2) for k, v in ev.items()},
        "event_sources": dict(by_src), "delivery_spans": dict(dlv),
        "delivery_per_hour": {k: round(v / hours, 2) for k, v in dlv.items()},
        "laughs_over_speech": laugh_over,
        "emotion_spans": dict(emo), "emotion_segments": dict(Counter(r["emotion_segment"] for r in rows)),
        "fillers": fill, "fillers_per_hour": round(fill / hours, 1),
        "audio_version": dict(Counter(r["audio_version"] for r in rows)),
        "mos": pct(q("mos"), (10, 25, 50, 75, 90)), "dnsmos": pct(q("dnsmos"), (10, 25, 50, 75, 90)),
        "mos_hist": np.histogram(q("mos"), bins=np.arange(1, 5.01, 0.1))[0].tolist(),
        "dnsmos_hist": np.histogram(q("dnsmos"), bins=np.arange(1, 5.01, 0.1))[0].tolist(),
        "asr_looped": sum(bool(r["asr_looped"]) for r in rows),
        "squeezed": {"segments": sum((r.get("squeezed_words") or 0) >= 2 for r in rows),
                     "hours": h([r["duration"] for r in rows if (r.get("squeezed_words") or 0) >= 2])},
        "edge_clipped": sum(bool(r["edge_clip_start"] or r["edge_clip_end"]) for r in rows),
        "emilia_filter": {"segments": len(emilia), "hours": h([r["duration"] for r in emilia])},
        "windowed": {"recordings": len(windowed), "hours": h([r["duration"] for r in rows if r["recording"] in windowed])},
        "examples": [{"id": r["id"], "tagged_text": r["tagged_text"]} for r in rich[:10]],
        "sound_tags": dict(snd.most_common()), "sound_tags_per_hour": {k: round(v / hours, 2) for k, v in snd.items()},
        "rows_with_sound_tags": snd_rows,
        "sounds": sounds_stats(rows, h),
    })


def sounds_stats(rows, h):
    """The `sounds` column: how many rows carry it, the labels heard at 0.5 or more, and the rows with music in them,
    by the audio version they are exported in."""
    have = [r for r in rows if r.get("sounds") is not None]
    heard = Counter(s["label"] for r in have for s in r["sounds"] if s["p"] >= 0.5)
    music = [r for r in have if any(s["label"] == "Music" and s["p"] >= 0.5 for s in r["sounds"])]
    return {"rows": len(have), "heard_at_0.5": dict(heard.most_common(30)), "music_rows": len(music),
            "music_hours": h([r["duration"] for r in music]),
            "music_by_audio_version": dict(Counter(r["audio_version"] for r in music))}


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block", type=int, default=64, help="recordings per shard (fixed: changing it re-cuts every shard)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--final", action="store_true", help="the run is over: pack every block, leaving out what failed")
    args = ap.parse_args()
    srcs = [json.loads(l) for l in args.sources.open()]
    blocks = [srcs[i:i + args.block] for i in range(0, len(srcs), args.block)]
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    done = lambda r: (args.work / r["id"] / "fill.jsonl").exists()  # noqa: E731
    # the vocal sounds' AudioSet pass (nonverbal/audioset_pass.py, which box_full_publish.sh runs first) is not one of
    # the run's stages: a block it has not reached waits, since a shard is written once and would ship without them
    tagged = lambda r: done(r) and (args.work / r["id"] / "audioset.jsonl").exists()  # noqa: E731
    empty = lambda r: r.get("bytes", EMPTY) < EMPTY  # noqa: E731
    todo = [b for b, recs in enumerate(blocks) if not shard_path(args.out, b).exists()
            and (args.final or all(tagged(r) or empty(r) for r in recs))]
    print(f"[hf] {len(blocks)} blocks of {args.block}; {len(blocks) - len(todo)} packed or waiting, packing {len(todo)}",
          flush=True)
    total = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (b, n, size) in enumerate(ex.map(pack, [(b, blocks[b], str(args.work), str(args.out)) for b in todo]), 1):
            total += size
            if k % 10 == 0 or k == len(todo):
                print(f"[hf] {k}/{len(todo)} blocks packed, {total / 2 ** 30:.1f} GiB", flush=True)
    from melati.stage_export import DEFAULTS
    (args.out / "export_config.json").write_text(json.dumps(DEFAULTS, indent=1))
    packed = [b for b in range(len(blocks)) if shard_path(args.out, b).exists()]
    recs = [r for b in packed for r in blocks[b]]
    sep = lambda r: args.work / r["id"] / "sep.json"  # noqa: E731
    dur_h = lambda rs: sum(json.loads(sep(r).read_text())["duration_s"] for r in rs if sep(r).exists()) / 3600  # noqa: E731
    src = {"source": SOURCE, "source_recordings": len(srcs), "blocks": len(blocks), "blocks_packed": len(packed),
           "complete": len(packed) == len(blocks),
           "release_recordings": len(recs), "release_source_hours": round(dur_h(recs), 1),
           "empty_source_files": sum(empty(r) for r in srcs),
           "failed_recordings": [r["id"] for r in recs if not done(r) and not empty(r)]}
    cols = [c for c in features() if c != "audio"]
    files = [shard_path(args.out, b) for b in packed]
    rows = [r for f in files for r in pq.read_table(f, columns=cols).to_pylist()]
    left = defaultdict(lambda: [0, 0.0])
    for b in packed:
        for why, (n, sec) in (json.loads(left_out_path(args.out, b).read_text())
                              if left_out_path(args.out, b).exists() else {}).items():
            left[why][0] += n
            left[why][1] += sec
    src["left_out"] = {k: {"segments": n, "hours": round(sec / 3600, 2)} for k, (n, sec) in sorted(left.items())}
    st = dict(stats(rows, src), shards=len(files), parquet_gib=round(sum(f.stat().st_size for f in files) / 2 ** 30, 1))
    (args.out / "stats.json").write_text(json.dumps(st, indent=1))
    print(f"[hf] {json.dumps({k: st[k] for k in ('blocks_packed', 'segments', 'hours', 'speakers', 'parquet_gib')})}")
    print("HF_BUILD_DONE")


if __name__ == "__main__":
    main()
