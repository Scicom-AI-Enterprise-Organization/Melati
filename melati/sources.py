"""List the source recordings: one jsonl row {id, path, stem, bytes} per audio file.

The podcast source is a 12-part `zip -s` split set; deploy/box_fetch_podcast.sh extracts it with
7-Zip (Python's zipfile cannot read a multi-disk archive) and this lists the extracted files in
place. File names are raw YouTube titles, full of spaces, quotes and emoji, so a recording's id is
a short hash of its path relative to the root; `stem` keeps the original name, which is what the
earlier Emilia outputs of the same recordings were named after and what the before/after
comparison joins on.

    python -m melati.sources --dir $MELATI_DATA/podcast/raw --list $MELATI_DATA/podcast/sources.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

AUDIO = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mp4")


def rec_id(name: str) -> str:
    return "r" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="the extracted source tree")
    ap.add_argument("--list", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for p in sorted(args.dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO:
            rel = str(p.relative_to(args.dir))
            rows.append({"id": rec_id(rel), "path": str(p), "stem": p.stem, "member": rel, "bytes": p.stat().st_size})
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "recording id collision"
    tmp = args.list.with_name(args.list.name + ".part")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.list)
    print(f"{len(rows)} recordings, {sum(r['bytes'] for r in rows) / 1e9:.1f} GB -> {args.list}")


if __name__ == "__main__":
    main()
