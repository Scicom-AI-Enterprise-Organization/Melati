#!/usr/bin/env python3
"""Melati against upstream Emilia-Pipe, on the same recordings. RUNS ON THE BOX.

Both pipelines process one set of recordings (scripts/make_pilot.py for the podcasts; the
VoxConverse dev set, whose reference speaker turns make purity measurable). This reads both
outputs and measures the same things on each, from the exported audio and its times, so no
number rests on either pipeline's account of itself:

  yield        segments and hours kept, as a share of the input, and their length; a pile-up
               near 30 s is a hard cut
  audio        codec, bitrate, spectral cutoff, level spread and clipping (scripts/audit_audio.py)
  edges        how often a segment's first or last WORD touches the cut: each pipeline's own
               transcript force-aligned to its own audio (align server), the first word starting or
               the last ending within 20 ms of the edge. The energy version (first or last 30 ms within
               12 dB of the speech level) is kept beside it, but it also fires on background noise,
               which a pipeline that always exports a separated stem has less of
  purity       (--refs) the share of a segment's reference speech in which only its main speaker
               talks, how often a second speaker or an overlap is inside it, and how much of the
               reference single-speaker speech each pipeline kept
  non-verbal   (--work + --endpoints) the laughs, coughs, breaths ... the detector finds in the
               ORIGINAL recording: how many fall inside a kept segment, how many are still
               detected in the exported audio (a separator can remove them), and how many
               Melati tagged
  code-switch  (wordfreq) English words inside Malay segments; a decode forced into one
               language translates them away
  throughput   (--*-timing) wall-clock seconds per hour of audio, and on how many GPUs

Melati filters by query, not before export, so it is read twice: everything it exported, and
the subset upstream's own filter would have kept (`emilia_filter`, a copy of
utils/tool.py:calculate_audio_stats). Upstream writes <input>_processed/<id>/<id>.json and
<id>_<i>.mp3; pass the `_processed` folder as --emilia.

    python scripts/compare_emilia.py --sources $MELATI_DATA/dev/vox/sources.jsonl \\
        --melati $MELATI_DATA/dev/vox/export --emilia $MELATI_DATA/dev/vox/emilia_in_processed \\
        --refs $MELATI_DATA/dev/vox --work $MELATI_DATA/dev/vox/work2 \\
        --endpoints $MELATI_DATA/serve_eval/endpoints.json --out results/compare/voxconverse.json
"""
import argparse, asyncio, json, random, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from melati import nonverbal as NV  # noqa: E402
from scripts.audit_audio import cutoff_hz, decode, ffprobe, loudness, ltas_db  # noqa: E402

FRAME_S = 0.01
EDGE_S = 0.03
HOT_DB = -12.0          # an edge frame this close to the segment's speech level is inside a sound
STEP = 0.01             # purity frame
TAG_GROUP = {v: k for k, v in NV.TO_TAG.items()}
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


# ---------------------------------------------------------------- reading both outputs

def emilia_segments(root: Path, rec_id: str):
    j = root / rec_id / f"{rec_id}.json"
    if not j.exists():
        return None
    return [{"start": float(r["start"]), "end": float(r["end"]), "text": r.get("text") or "",
             "language": r.get("language"), "dnsmos": r.get("dnsmos"),
             "audio": root / rec_id / f"{rec_id}_{i}.mp3"} for i, r in enumerate(json.loads(j.read_text()))]


def melati_segments(export: Path):
    import pyarrow.parquet as pq
    by = defaultdict(list)
    for f in sorted((export / "manifest").glob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            by[r["recording"]].append({"start": r["start"], "end": r["end"], "text": r["text"] or "",
                                       "language": r["language"], "dnsmos": r["dnsmos"],
                                       "audio": export / r["audio"], "annotation": r["annotation"]})
    return by


def char_count(text):
    return len(re.sub(r"[,.!?\"'，。！？“”‘’ ]", "", text))            # upstream get_char_count


def emilia_filter(segs, min_d=3, max_d=30, min_dnsmos=3, min_chars=2):
    """Upstream's post-ASR filter (utils/tool.py:calculate_audio_stats), applied per recording."""
    per = [(s["end"] - s["start"]) / char_count(s["text"]) for s in segs if char_count(s["text"]) > 0]
    if per:
        q1, q3 = np.percentile(per, 25), np.percentile(per, 75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    else:
        lo, hi = 0, np.inf
    out = []
    for s in segs:
        d, c = s["end"] - s["start"], char_count(s["text"])
        if (min_d <= d <= max_d and (s["dnsmos"] or 0) >= min_dnsmos and c >= min_chars
                and lo <= (d / c if c else 0) <= hi):
            out.append(s)
    return out


def duration_s(rec):
    if rec.get("duration_s"):
        return float(rec["duration_s"])
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          rec["path"]], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def pct(v, qs):
    v = [a for a in v if a is not None]
    return {str(q): (round(float(np.percentile(v, q)), 3) if v else None) for q in qs}


def share(v):
    v = list(v)
    return round(float(np.mean(v)), 4) if v else None


# ---------------------------------------------------------------- measures

def yield_of(by_rec, input_s):
    segs = [s for r in input_s for s in by_rec.get(r) or []]
    d = [s["end"] - s["start"] for s in segs]
    tot_in = sum(input_s.values())
    return {"recordings_with_output": sum(1 for r in input_s if by_rec.get(r)), "segments": len(segs),
            "hours": round(sum(d) / 3600, 3), "share_of_input": round(sum(d) / tot_in, 4) if tot_in else None,
            "duration_s": pct(d, (10, 50, 90, 99)), "share_ge_29_5s": share(x >= 29.5 for x in d),
            "share_lt_3s": share(x < 3 for x in d),
            "languages": dict(Counter(s["language"] for s in segs).most_common(8))}


def edge_hot(x, sr):
    n = int(FRAME_S * sr)
    m = len(x) // n
    if m < 20:
        return None
    db = 10 * np.log10(np.mean(x[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    level = float(np.median(db[db >= db.max() - 40]))
    k = max(1, int(round(EDGE_S / FRAME_S)))
    return bool(db[:k].max() >= level + HOT_DB), bool(db[-k:].max() >= level + HOT_DB)


def audio_of(segs, sample, rng):
    pick = rng.sample(segs, min(sample, len(segs)))
    rows = []
    for s in pick:
        p = Path(s["audio"])
        if not p.exists():
            continue
        data = p.read_bytes()
        codec, _, br = ffprobe(data)
        x, sr = decode(data)
        if len(x) < 2048:
            continue
        f, db = ltas_db(x, sr)
        peak = float(np.max(np.abs(x)))
        e = edge_hot(x, sr)
        rows.append({"codec": codec, "sr": sr, "bitrate": br, "cutoff_hz": cutoff_hz(f, db),
                     "lufs": loudness(x, sr), "peak_db": 20 * np.log10(peak) if peak > 0 else None,
                     "full_scale": peak >= 10 ** (-0.1 / 20), "over_full_scale": peak > 1.0,
                     "len_err_ms": 1000 * (len(x) / sr - (s["end"] - s["start"])),
                     "edge_start": e[0] if e else None, "edge_end": e[1] if e else None})
    br = [r["bitrate"] / 1000 for r in rows if r["bitrate"]]
    lufs = [r["lufs"] for r in rows if r["lufs"] is not None]
    return {"files": len(rows), "codecs": dict(Counter(r["codec"] for r in rows)),
            "sample_rates": dict(Counter(str(r["sr"]) for r in rows)), "bitrate_kbps": pct(br, (10, 50, 90)),
            "cutoff_khz": pct([r["cutoff_hz"] / 1000 for r in rows], (10, 50, 90)),
            "lufs": pct(lufs, (5, 50, 95)),
            "lufs_iqr": round(float(np.percentile(lufs, 75) - np.percentile(lufs, 25)), 2) if lufs else None,
            "peak_db": pct([r["peak_db"] for r in rows], (5, 50, 95)),
            "share_peak_at_full_scale": share(r["full_scale"] for r in rows),
            "share_over_full_scale": share(r["over_full_scale"] for r in rows),
            "decoded_minus_stated_ms": pct([r["len_err_ms"] for r in rows], (5, 50, 95)),
            "share_start_inside_sound": share(r["edge_start"] for r in rows if r["edge_start"] is not None),
            "share_end_inside_sound": share(r["edge_end"] for r in rows if r["edge_end"] is not None)}


def ref_frames(turns):
    spk = sorted({t[2] for t in turns})
    n = int(max([e for _, e, _ in turns] + [0]) / STEP) + 2
    m = np.zeros((len(spk), n), bool)
    for s, e, k in turns:
        m[spk.index(k), int(s / STEP):int(e / STEP)] = True
    return m


def purity_of(by_rec, refs):
    per, kept_single, all_single = [], 0, 0
    for rec, R in refs.items():
        segs = by_rec.get(rec) or []
        n = R.shape[1]
        active = R.sum(0)
        single = active == 1
        cover = np.zeros(n, bool)
        for s in segs:
            a, b = int(s["start"] / STEP), min(n, int(s["end"] / STEP))
            if b <= a:
                continue
            cover[a:b] = True
            sub = R[:, a:b]
            voiced = sub.any(0)
            v = int(voiced.sum())
            if v == 0:
                per.append({"dur": (b - a) * STEP, "voiced": 0.0})
                continue
            spk_s = sub.sum(1)
            main = int(np.argmax(spk_s))
            only_main = sub[main] & (sub.sum(0) == 1)
            others = np.delete(spk_s, main)
            per.append({"dur": (b - a) * STEP, "voiced": v / (b - a), "purity": float(only_main.sum() / v),
                        "overlap_s": float((sub.sum(0) >= 2).sum() * STEP),
                        "second_s": float(others.max() * STEP) if len(others) else 0.0})
        kept_single += int((single & cover).sum())
        all_single += int(single.sum())
    sp = [p for p in per if "purity" in p]
    w = np.array([p["dur"] for p in sp])
    return {"segments": len(per),
            "purity_weighted": round(float(np.sum(w * [p["purity"] for p in sp]) / w.sum()), 4) if sp else None,
            "share_purity_lt_0_9": share(p["purity"] < 0.9 for p in sp),
            "share_second_speaker_ge_0_5s": share(p["second_s"] >= 0.5 for p in sp),
            "share_overlap_ge_0_2s": share(p["overlap_s"] >= 0.2 for p in sp),
            "share_mostly_no_reference_speech": share(p["voiced"] < 0.5 for p in per),
            "single_speaker_speech_kept": round(kept_single / all_single, 4) if all_single else None}


def wordlang():
    try:
        from wordfreq import zipf_frequency
        zipf_frequency("makan", "ms")
    except Exception as e:                       # noqa: BLE001
        print(f"code-switch: skipped ({str(e)[:80]})", flush=True)
        return None

    def lang(w):
        en, ms = zipf_frequency(w, "en"), zipf_frequency(w, "ms")
        if en >= 3.0 and en >= ms + 1.0:
            return "en"
        if ms >= 3.0 and ms >= en + 1.0:
            return "ms"
        return None
    return lang


def code_switch_of(segs, lang):
    """Over segments whose classified words are mostly Malay: the share of those words that are
    English, and the share of segments with at least two of each."""
    en_in_ms, words_in_ms, mixed, ms_segs = 0, 0, 0, 0
    for s in segs:
        tags = [lang(w.lower()) for w in WORD.findall(s["text"])]
        n_en, n_ms = tags.count("en"), tags.count("ms")
        if n_ms < 2 or n_ms < n_en:
            continue
        ms_segs += 1
        en_in_ms += n_en
        words_in_ms += n_en + n_ms
        mixed += n_en >= 2
    return {"malay_segments": ms_segs, "english_word_share": round(en_in_ms / words_in_ms, 4) if words_in_ms else None,
            "share_code_switched": round(mixed / ms_segs, 4) if ms_segs else None}


async def clipped_words_of(pipes, sample, endpoints, edge_s=0.02, seed=0):
    """Per pipeline: align its own transcript to its own exported audio; a segment is clipped at the
    start when its first word begins within edge_s of the cut, at the end likewise."""
    from melati.client import Client, Endpoints
    out = {}
    async with Client(Endpoints(endpoints)) as c:
        for name, segs in pipes.items():
            rng = random.Random(seed)
            pick = [s for s in rng.sample(segs, min(sample, len(segs))) if s["text"].strip() and Path(s["audio"]).exists()]

            async def one(s):
                x, sr = decode(Path(s["audio"]).read_bytes())
                x16 = A.resample(x, sr, 16000)
                if len(x16) > 30 * 16000 or len(x16) < 1600:
                    return None
                lang = (s.get("language") or "")[:2] or None
                r = (await c.align([(x16, s["text"], lang)]))[0]
                ws = r.get("words") or []
                if not ws:
                    return None
                dur = len(x16) / 16000
                return ws[0]["s"] <= edge_s, ws[-1]["e"] >= dur - edge_s
            res = [r for r in await asyncio.gather(*[one(s) for s in pick]) if r is not None]
            out[name] = {"aligned": len(res), "of": len(pick),
                         "share_first_word_at_cut": share(a for a, _ in res),
                         "share_last_word_at_cut": share(b for _, b in res),
                         "share_either": share(a or b for a, b in res)}
            print(f"[edges] {name}: {out[name]}", flush=True)
    return out


async def nonverbal_of(pipes, recs, work: Path, endpoints, floor: float):
    """Events found in the ORIGINAL recording's stored SED tracks (the mix, before separation),
    followed into each pipeline's exported audio."""
    from melati.client import Client, Endpoints
    src = {}
    for rec in recs:
        f = work / rec / "sed.npz"
        if not f.exists():
            continue
        tr = dict(np.load(f))
        fps = int(tr.pop("fps"))
        dur = min(len(v) for v in tr.values()) / fps
        src[rec] = [e for e in NV.events_from_tracks(tr, fps, 0.0, dur) if e["label"] in NV.TO_TAG and e["p"] >= floor]
    out = {"recordings": len(src), "floor": floor,
           "source_events": dict(Counter(e["label"] for ev in src.values() for e in ev))}
    async with Client(Endpoints(endpoints)) as c:
        for name, by_rec in pipes.items():
            jobs = []
            for rec, evs in src.items():
                for s in by_rec.get(rec) or []:
                    hit = [e for e in evs if min(e["e"], s["end"] - s["start"] + s["start"]) - max(e["s"], s["start"]) > 0]
                    if hit and Path(s["audio"]).exists():
                        jobs.append((rec, s, hit))

            async def redetect(rec, s, hit):
                x, sr = decode(Path(s["audio"]).read_bytes())
                tr = await c.sed(A.resample(x, sr, 32000))
                fps = int(tr.pop("fps"))
                found = NV.events_from_tracks(tr, fps, 0.0, len(x) / sr)
                return rec, s, hit, found
            res = await asyncio.gather(*[redetect(*j) for j in jobs])
            inside, audible, tagged = Counter(), Counter(), Counter()
            seen = set()
            for rec, s, hit, found in res:
                tags = []
                if s.get("annotation"):
                    tags = [(TAG_GROUP.get(e["kind"]), e["start"] + s["start"], e["end"] + s["start"])
                            for e in json.loads(s["annotation"]).get("events") or []]
                for e in hit:
                    key = (rec, e["label"], e["s"])
                    covered = min(e["e"], s["end"]) - max(e["s"], s["start"])
                    if key in seen or covered < 0.5 * (e["e"] - e["s"]):
                        continue
                    seen.add(key)
                    inside[e["label"]] += 1
                    if any(f["label"] == e["label"] and min(f["e"] + s["start"], e["e"] + 0.25)
                           - max(f["s"] + s["start"], e["s"] - 0.25) > 0 for f in found):
                        audible[e["label"]] += 1
                    if any(g == e["label"] and min(b, e["e"] + 0.25) - max(a, e["s"] - 0.25) > 0 for g, a, b in tags):
                        tagged[e["label"]] += 1
            n = out["source_events"]
            out[name] = {cls: {"inside_kept": round(inside[cls] / n[cls], 4), "still_audible": round(audible[cls] / n[cls], 4),
                               **({"tagged": round(tagged[cls] / n[cls], 4)} if name.startswith("melati") else {})}
                         for cls in n}
            print(f"[nonverbal] {name}: {sum(inside.values())} inside, {sum(audible.values())} audible", flush=True)
    return out


def timing_of(path, gpus, hours):
    if not path or not path.exists():
        return None
    if path.suffix == ".jsonl":
        wall = sum(json.loads(l)["wall_s"] for l in path.open() if l.strip())
    else:
        wall = float(json.loads(path.read_text())["wall_s"])
    return {"wall_s": round(wall, 1), "gpus": gpus, "x_realtime": round(hours * 3600 / wall, 1),
            "x_realtime_per_gpu": round(hours * 3600 / wall / gpus, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--melati", type=Path, required=True, help="Melati export dir (manifest/ + audio/)")
    ap.add_argument("--emilia", type=Path, required=True, help="upstream's <input>_processed dir")
    ap.add_argument("--refs", type=Path, default=None, help="dir of <id>.ref.json reference turns")
    ap.add_argument("--work", type=Path, default=None, help="Melati work dir (sed.npz per recording)")
    ap.add_argument("--endpoints", nargs="+", type=Path, default=None,
                    help="align (edges) + front (events in exported audio); several files merge")
    ap.add_argument("--nv-floor", type=float, default=0.3)
    ap.add_argument("--audio-sample", type=int, default=1500)
    ap.add_argument("--melati-timing", type=Path, default=None, help="run.py --timing jsonl of one full pass")
    ap.add_argument("--melati-gpus", type=int, default=1)
    ap.add_argument("--emilia-timing", type=Path, default=None, help='{"wall_s": ...}')
    ap.add_argument("--emilia-gpus", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.sources.open() if l.strip()]
    em = {r["id"]: emilia_segments(args.emilia, r["id"]) for r in recs}
    done = [r for r in recs if em[r["id"]] is not None]
    print(f"{len(recs)} recordings, upstream finished {len(done)}", flush=True)
    input_s = {r["id"]: duration_s(r) for r in done}
    input_s = {k: v for k, v in input_s.items() if v}
    mel = melati_segments(args.melati)
    pipes = {"emilia": {k: v for k, v in em.items() if v is not None},
             "melati_all": dict(mel),
             "melati_emilia_filter": {k: emilia_filter(v) for k, v in mel.items()}}
    hours = sum(input_s.values()) / 3600
    out = {"recordings": len(input_s), "input_hours": round(hours, 3),
           "not_processed_by_emilia": [r["id"] for r in recs if em[r["id"]] is None], "pipelines": {}}
    rng = random.Random(0)
    lang = wordlang()
    refs = {}
    if args.refs:
        for rec in input_s:
            f = args.refs / f"{rec}.ref.json"
            if f.exists():
                refs[rec] = ref_frames(json.loads(f.read_text())["turns"])
    for name, by_rec in pipes.items():
        segs = [s for r in input_s for s in by_rec.get(r) or []]
        o = {"yield": yield_of(by_rec, input_s)}
        o["audio"] = audio_of(segs, args.audio_sample, rng)
        if refs:
            o["purity"] = purity_of(by_rec, refs)
        if lang:
            o["code_switch"] = code_switch_of(segs, lang)
        out["pipelines"][name] = o
        print(f"[{name}] {json.dumps(o['yield'])[:220]}", flush=True)
    if args.endpoints:
        out["edges_by_alignment"] = asyncio.run(clipped_words_of(
            {k: [s for r in input_s for s in p.get(r) or []] for k, p in pipes.items()},
            args.audio_sample, args.endpoints))
    if args.work and args.endpoints:
        out["nonverbal"] = asyncio.run(nonverbal_of({k: {r: v for r, v in p.items() if r in input_s}
                                                     for k, p in pipes.items() if k != "melati_emilia_filter"},
                                                    list(input_s), args.work, args.endpoints, args.nv_floor))
    out["throughput"] = {"melati": timing_of(args.melati_timing, args.melati_gpus, hours),
                         "emilia": timing_of(args.emilia_timing, args.emilia_gpus, hours)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "pipelines"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
