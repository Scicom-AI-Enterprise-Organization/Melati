#!/usr/bin/env python3
"""The sounds stage (melati.pipeline.sounds) on a few hours of the full run before it joins the run: which
recordings, and what the rows' `sounds` column then says. RUNS ON THE BOX, in .venv-melati.

    python scripts/sounds_pilot.py pick --hours 4 --out $MELATI_DATA/podcast/sounds_pilot/sources.jsonl
    python scripts/sounds_pilot.py report --sources $MELATI_DATA/podcast/sounds_pilot/sources.jsonl \\
        --timing $MELATI_DATA/podcast/sounds_pilot/timing.jsonl --out $MELATI_DATA/results/sounds/pilot.json

    python scripts/sounds_pilot.py compare --sources .../sources.jsonl --work $MELATI_DATA/podcast/work \\
        $MELATI_DATA/podcast/sounds_pilot50_ens/work --names v4 ensemble --out .../pilot50_compare.json

`pick` draws finished recordings (fill.jsonl written) at random, seed 0, until they hold --hours of audio. `report`
builds their rows as the dataset would (melati.pipeline.export_rows, no audio) and counts the labels: per label the
rows that carry it at each threshold and the hours they hold, by AudioSet's top-level categories, and how many labels
a row carries. `compare` sets two taggers' stage files on the same recordings side by side, row by row. The results
files hold counts only; a few rows are printed to the log as examples, never saved.
"""
import os
import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PODCAST = Path(f"{MELATI_DATA}/podcast")
AUDIOSET = Path(f"{MELATI_DATA}/audioset")
THRESHOLDS = (0.1, 0.3, 0.5, 0.7)
SPEECH = "/m/09x0r"          # AudioSet's Speech: it and the classes under it are on nearly every row


def pick(a):
    recs = [json.loads(line) for line in (PODCAST / "sources.jsonl").open()]
    random.Random(0).shuffle(recs)
    out, hours = [], 0.0
    for r in recs:
        d = PODCAST / "work" / r["id"]
        if not ((d / "fill.jsonl").exists() and (d / "vocal16k.flac").exists()):
            continue
        out.append(r)
        hours += json.loads((d / "sep.json").read_text())["duration_s"] / 3600
        if hours >= a.hours:
            break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"[pick] {len(out)} recordings, {hours:.2f} h -> {a.out}")


def categories():
    """Each class name -> the first of the ontology's top-level categories that reaches it (as the model card
    groups them), and the names under Speech."""
    nodes = json.loads((AUDIOSET / "ontology.json").read_text())
    by_id = {n["id"]: n for n in nodes}
    child = {c for n in nodes for c in n["child_ids"]}
    with (AUDIOSET / "class_labels_indices.csv").open() as fh:
        name = {row["mid"]: row["display_name"] for row in csv.DictReader(fh)}

    def under(top):
        found, todo = set(), [top]
        while todo:
            m = todo.pop()
            found.add(m)
            todo += by_id[m]["child_ids"]
        return found
    cat = {}
    for n in nodes:
        if n["id"] not in child:
            for m in under(n["id"]):
                if m in name:
                    cat.setdefault(name[m], n["name"])
    return cat, {name[m] for m in under(SPEECH) if m in name}


def report(a):
    from melati.pipeline import export_rows
    from melati.stage_export import DEFAULTS
    recs = [json.loads(line) for line in a.sources.open()]
    cat, speech = categories()
    rows, windows, stored, model = [], Counter(), 0, None
    for r in recs:
        rows += [row for row, _ in export_rows(a.work, r, {}, audio=False)]
        for line in (a.work / r["id"] / "sounds.jsonl").open():
            s = json.loads(line)
            windows[s["windows"]] += 1
            stored += len(s["p"])
            model = s["model"]
    rows = [r for r in rows if r["sounds"] is not None]
    hours = sum(r["duration"] for r in rows) / 3600
    per = {}
    for r in rows:
        for s in r["sounds"]:
            e = per.setdefault(s["label"], {t: [0, 0.0] for t in THRESHOLDS})
            for t in THRESHOLDS:
                if s["p"] >= t:
                    e[t][0] += 1
                    e[t][1] += r["duration"]
    labels = {k: {"category": cat.get(k), "speech": k in speech,
                  **{f"rows_p{t}": v[t][0] for t in THRESHOLDS},
                  **{f"share_p{t}": round(v[t][0] / len(rows), 4) for t in THRESHOLDS},
                  **{f"hours_p{t}": round(v[t][1] / 3600, 3) for t in THRESHOLDS}}
              for k, v in sorted(per.items(), key=lambda kv: -kv[1][0.5][0])}
    by_cat = {}
    for t in THRESHOLDS:
        for c in sorted(set(cat.values())):
            n = sum(any(s["p"] >= t and cat.get(s["label"]) == c and s["label"] not in speech for s in r["sounds"])
                    for r in rows)
            by_cat.setdefault(c, {})[f"rows_p{t}"] = n
    count = {t: [sum(s["p"] >= t for s in r["sounds"]) for r in rows] for t in THRESHOLDS}
    beyond = {t: [sum(s["p"] >= t and s["label"] not in speech for s in r["sounds"]) for r in rows]
              for t in THRESHOLDS}
    # music the tagger hears in the audio a row ships: which version that is, and how far under the voice the
    # front stage measured the music (None where it separated nothing, so measured none)
    music = {}
    for name, sel in (("music", lambda r: any(s["p"] >= 0.5 and cat.get(s["label"]) == "Music" for s in r["sounds"])),
                      ("rest", lambda r: not any(s["p"] >= 0.5 and cat.get(s["label"]) == "Music" for s in r["sounds"]))):
        sub = [r for r in rows if sel(r)]
        bg = [r["background_db"] for r in sub if r["background_db"] is not None]
        music[name] = {"rows": len(sub), "audio_version": dict(Counter(r["audio_version"] for r in sub)),
                       "background_db_measured": len(bg),
                       "background_db_median": round(statistics.median(bg), 1) if bg else None}
    # the point tags the export writes from the same stage ([music], [applause]; melati.stage_export.sound_events)
    inline, rows_with = Counter(), 0
    for r in rows:
        kinds = [e["kind"] for e in json.loads(r["annotation"])["events"] if e.get("source") == "sounds"]
        inline.update(kinds)
        rows_with += bool(kinds)
    timing = [json.loads(line) for line in a.timing.open()] if a.timing and a.timing.exists() else []
    res = {"recordings": len(recs), "rows": len(rows), "hours": round(hours, 3), "model": model,
           "column_min_p": DEFAULTS["sounds_min_p"], "stored_per_segment": round(stored / max(sum(windows.values()), 1), 1),
           "windows_per_segment": dict(sorted(windows.items())),
           "labels_per_row": {f"p{t}": {"mean": round(statistics.mean(count[t]), 2),
                                        "median": statistics.median(count[t]), "max": max(count[t])}
                              for t in THRESHOLDS},
           "rows_with_a_sound_beyond_speech": {f"p{t}": sum(n > 0 for n in beyond[t]) for t in THRESHOLDS},
           "labels_heard": {f"p{t}": sum(v[f"rows_p{t}"] > 0 for v in labels.values()) for t in THRESHOLDS},
           "inline": {"rows": rows_with, "tags": sum(inline.values()),
                      "per_hour": round(sum(inline.values()) / max(hours, 1e-9), 2), "by_tag": dict(inline.most_common()),
                      "gates": DEFAULTS["sounds_inline_gates"], "vocabulary": len(DEFAULTS["sounds_inline"])},
           "music_vs_shipped_audio": music, "by_category": by_cat, "labels": labels,
           "stage": timing[-1] if timing else None}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("labels", "by_category")}, indent=1))
    print("[report] most frequent labels beyond speech, rows at p >= 0.5 (share):")
    for k, v in [kv for kv in labels.items() if not kv[1]["speech"]][:25]:
        print(f"  {v['rows_p0.5']:6d} ({100 * v['share_p0.5']:5.1f}%)  {k}  [{v['category']}]")
    print("[report] examples (log only):")
    seen = set()
    for want in ("Music", "Laughter", "Applause", "Telephone", "Cough", "Singing", "Television", "Vehicle"):
        best = sorted((r for r in rows if r["id"] not in seen),
                      key=lambda r: -max((s["p"] for s in r["sounds"] if s["label"] == want), default=0))[:2]
        for r in best:
            if not any(s["label"] == want and s["p"] >= 0.3 for s in r["sounds"]):
                continue
            seen.add(r["id"])
            top = ", ".join(f"{s['label']} {s['p']:.2f}" for s in r["sounds"][:6])
            print(f"  {r['id']} ({r['duration']:.1f} s) [{want}] {r['tagged_text'][:140]}\n      sounds: {top}")
    print(f"[report] inline sound tags: {res['inline']['tags']} in {rows_with} rows, {res['inline']['by_tag']}")
    print("[report] rows with inline sound tags (log only):")
    tagged = [r for r in rows if r.get("n_sound_tags")]
    other = [r for r in tagged if any(e.get("source") == "sounds" and e["kind"] != "music"
                                      for e in json.loads(r["annotation"])["events"])]
    for r in other + [r for r in tagged if r not in other][:12]:
        top = ", ".join(f"{s['label']} {s['p']:.2f}" for s in r["sounds"][:4])
        print(f"  {r['id']} ({r['duration']:.1f} s) {r['tagged_text'][:170]}\n      sounds: {top}")
    print(f"[report] -> {a.out}")


def compare(a):
    """Two taggers' sounds stage on the same recordings, row by row (matched by id): the inline tags each writes and
    where they agree, and per label beyond speech the rows each hears it in at p >= 0.5. Counts only."""
    from melati.pipeline import export_rows
    recs = [json.loads(line) for line in a.sources.open()]
    _, speech = categories()
    rows = [{row["id"]: row for r in recs for row, _ in export_rows(w, r, {}, audio=False)} for w in a.work]
    ids = sorted(set(rows[0]) & set(rows[1]))
    models = [json.loads(next((w / r["id"] / "sounds.jsonl").open()))["model"] for w in a.work for r in recs[:1]]

    def tags(row):
        return Counter(e["kind"] for e in json.loads(row["annotation"])["events"] if e.get("source") == "sounds")

    def heard(row, t=0.5):
        return {s["label"] for s in row["sounds"] or [] if s["p"] >= t and s["label"] not in speech}
    inline, per = {}, {}
    for i in ids:
        ta, tb = tags(rows[0][i]), tags(rows[1][i])
        for k in set(ta) | set(tb):
            e = inline.setdefault(k, {"rows_" + a.names[0]: 0, "rows_" + a.names[1]: 0, "rows_both": 0})
            e["rows_" + a.names[0]] += k in ta
            e["rows_" + a.names[1]] += k in tb
            e["rows_both"] += k in ta and k in tb
        ha, hb = heard(rows[0][i]), heard(rows[1][i])
        for k in ha | hb:
            e = per.setdefault(k, {"rows_" + a.names[0]: 0, "rows_" + a.names[1]: 0, "rows_both": 0})
            e["rows_" + a.names[0]] += k in ha
            e["rows_" + a.names[1]] += k in hb
            e["rows_both"] += k in ha and k in hb
    # where one tagger writes an inline tag and the other does not: each one's probability for that class there
    # (the `sounds` column, so under 0.1 reads 0), and per segment length the rows each hears a class in
    from melati.stage_export import SOUNDS_INLINE

    def p(row, cls):
        return next((s["p"] for s in row["sounds"] or [] if s["label"] == cls), 0.0)

    def quartiles(v):
        return [round(float(x), 3) for x in statistics.quantiles(v, n=4)] if len(v) > 1 else v
    cls_of = {v["tag"]: k for k, v in SOUNDS_INLINE["classes"].items()}
    only = {}
    for i in ids:
        ta, tb = tags(rows[0][i]), tags(rows[1][i])
        for k in set(ta) ^ set(tb):
            e = only.setdefault(f"{k} only by {a.names[0] if k in ta else a.names[1]}", {n: [] for n in a.names})
            for n, rs in zip(a.names, rows):
                e[n].append(p(rs[i], cls_of[k]))
    only = {k: {"rows": len(v[a.names[0]]), **{f"p_{n} quartiles": quartiles(x) for n, x in v.items()}}
            for k, v in sorted(only.items(), key=lambda kv: -len(kv[1][a.names[0]]))}
    edges = (0, 3, 6, 10, 1e9)
    by_len = {}
    for lo, hi in zip(edges, edges[1:]):
        sel = [i for i in ids if lo <= rows[0][i]["duration"] < hi]
        e = by_len[f"{lo}-{hi if hi < 1e9 else ''} s"] = {"rows": len(sel)}
        for cls, t in (("Music", 0.5), ("Music", 0.8), ("Silence", 0.1), ("Silence", 0.5)):
            for n, rs in zip(a.names, rows):
                e[f"{cls} p{t} {n}"] = sum(p(rs[i], cls) >= t for i in sel)
    side = {n: {"inline_tags": sum(sum(tags(rs[i]).values()) for i in ids),
                "rows_with_inline_tags": sum(bool(tags(rs[i])) for i in ids),
                "rows_with_a_sound_beyond_speech_p0.5": sum(bool(heard(rs[i])) for i in ids),
                "labels_per_row_p0.1": round(statistics.mean(len(rs[i]["sounds"] or []) for i in ids), 2)}
            for n, rs in zip(a.names, rows)}
    res = {"recordings": len(recs), "rows": len(ids), "rows_only_in_one": len(set(rows[0]) ^ set(rows[1])),
           "models": dict(zip(a.names, models)), "sides": side,
           "inline_by_tag": dict(sorted(inline.items(), key=lambda kv: -max(kv[1].values()))),
           "inline_where_they_differ": only, "by_segment_length": by_len,
           "labels_p0.5": dict(sorted(per.items(), key=lambda kv: -max(kv[1].values())))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "labels_p0.5"}, indent=1))
    print("[compare] labels beyond speech at p >= 0.5, rows (" + " / ".join(a.names) + " / both):")
    for k, v in list(res["labels_p0.5"].items())[:30]:
        print(f"  {' / '.join(str(x) for x in v.values()):>20}  {k}")
    print("[compare] rows where the inline tags differ (log only):")
    for i in [i for i in ids if tags(rows[0][i]) != tags(rows[1][i])][:30]:
        print(f"  {i} {a.names[0]} {dict(tags(rows[0][i]))} {a.names[1]} {dict(tags(rows[1][i]))}: "
              f"{rows[1][i]['tagged_text'][:150]}")
    print(f"[compare] -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--hours", type=float, default=4.0)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--work", type=Path, default=PODCAST / "work")
    p.add_argument("--timing", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--work", type=Path, nargs=2, required=True, help="the run's work dir, then the other tagger's")
    p.add_argument("--names", nargs=2, default=("a", "b"))
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"pick": pick, "report": report, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
