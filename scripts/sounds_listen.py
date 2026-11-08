#!/usr/bin/env python3
"""The sounds pilot's inline tags as clips to listen to (scripts/sounds_pilot.py's recordings): for every point tag the
export writes from the sounds stage, the row's own audio (the version the dataset ships) from BEFORE seconds ahead of
the tag to AFTER seconds past it, as MP3, a manifest of what a listener sees beside each clip, and one HTML file with
every clip inside it (scripts/sounds_check.template.html) where a listener marks each tag right or wrong and exports
the answers as JSON. It holds transcripts, so it stays on the box and with the listener, never in results/. RUNS ON
THE BOX, in .venv-melati.

    python scripts/sounds_listen.py --sources $MELATI_DATA/podcast/sounds_pilot/sources.jsonl \\
        --out $MELATI_DATA/podcast/sounds_pilot/listen [--per-tag 20]
"""
import os
import argparse
import base64
import json
import random
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melati.pipeline import SR24, export_rows  # noqa: E402
from melati.stage_export import DEFAULTS, SOUNDS_INLINE  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

PODCAST = Path(f"{MELATI_DATA}/podcast")
BEFORE, AFTER = 3.0, 7.0


def mp3(x, path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR24), "-ac", "1", "-i", "-",
                    "-b:a", "48k", "-y", str(path)], input=np.asarray(x, np.float32).tobytes(), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-tag", type=int, default=0, help="at most this many clips of one tag, drawn at random "
                                                            "(seed 0); 0 keeps them all")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    cls_of = {v["tag"]: k for k, v in SOUNDS_INLINE["classes"].items()}
    recs = [json.loads(line) for line in a.sources.open()]
    # every tag first, without audio and in parallel; then the clips of the ones kept, decoding only the recordings
    # that hold one
    with ProcessPoolExecutor(a.workers) as pool:
        found = [t for ts in pool.map(tags_of, recs, chunksize=8) for t in ts]
    by_tag = Counter(kind for _, _, kind in found)
    keep = set()
    for tag in by_tag:
        ids = [cid for _, cid, kind in found if kind == tag]
        keep |= set(random.Random(0).sample(ids, a.per_tag) if a.per_tag and len(ids) > a.per_tag else ids)
    holding = {rid for rid, cid, _ in found if cid in keep}
    items = []
    for rec in (r for r in recs if r["id"] in holding):
        for row, x in export_rows(PODCAST / "work", rec, {}, audio=True):
            evs = [e for e in json.loads(row["annotation"])["events"] if e.get("source") == "sounds"]
            for k, e in enumerate(evs):
                cid = f"{row['id']}_{k}"
                if cid not in keep:
                    continue
                s, t = max(0.0, e["start"] - BEFORE), min(row["duration"], e["start"] + AFTER)
                mp3(x[int(s * SR24):int(t * SR24)], a.out / "clips" / f"{cid}.mp3")
                cls = cls_of[e["kind"]]
                items.append({"id": cid, "row": row["id"], "recording": row["recording"], "tag": e["kind"],
                              "class": cls, "onset": e["start"], "end": e["end"], "clip": [round(s, 3), round(t, 3)],
                              "frames_peak": e["conf"], "threshold": DEFAULTS["sounds_inline"][cls],
                              "v4": next((v["p"] for v in row["sounds"] if v["label"] == cls), None),
                              "sounds": row["sounds"][:5], "tagged_text": row["tagged_text"],
                              "duration": row["duration"], "audio_version": row["audio_version"],
                              "background_db": row["background_db"], "language": row["language"]})
    hours = sum(json.loads((PODCAST / "work" / r["id"] / "sep.json").read_text())["duration_s"] for r in recs) / 3600
    meta = {"recordings": len(recs), "hours": round(hours, 1), "rows": len({cid.rsplit("_", 1)[0] for _, cid, _ in found}),
            "tags": dict(by_tag.most_common()), "shown": dict(Counter(i["tag"] for i in items))}
    (a.out / "manifest.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False))
    page(items, meta, a.out)
    print(f"[listen] {len(items)} clips of {len({i['row'] for i in items})} rows -> {a.out}; {meta}", flush=True)


def tags_of(rec):
    """One recording's sound tags as (recording, clip id, tag), from its rows as the dataset holds them."""
    return [(rec["id"], f"{row['id']}_{k}", e["kind"])
            for row, _ in export_rows(PODCAST / "work", rec, {}, audio=False)
            for k, e in enumerate(ev for ev in json.loads(row["annotation"])["events"] if ev.get("source") == "sounds")]


def page(items, meta, out):
    """The check as one HTML file, every clip inside it as a data URI, so it opens anywhere without a server."""
    keep = ("id", "row", "tag", "class", "onset", "end", "clip", "frames_peak", "threshold", "v4", "tagged_text",
            "audio_version")
    rows = [dict({k: it[k] for k in keep}, sounds=[{"label": s["label"], "p": s["p"]} for s in it["sounds"][:4]],
                 audio="data:audio/mpeg;base64,"
                 + base64.b64encode((out / "clips" / f"{it['id']}.mp3").read_bytes()).decode()) for it in items]
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = (Path(__file__).with_name("sounds_check.template.html")).read_text()
    (out / "melati-sound-check.html").write_text(tpl.replace("__META__", json.dumps(meta)).replace("__DATA__", data))


if __name__ == "__main__":
    main()
