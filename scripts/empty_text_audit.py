"""How often the ASR came back with no text, hour by hour, to tell the model hearing nothing from a server failing.

vLLM's transcription route answers an exception it did not expect with HTTP 200 and an {"error": ...} body, and
until 2026-10-26 the client read that as a transcript with no text. The one such exception the run has met is
benign (a decode that is only end-of-text), but a replica that failed every request the same way would have
blanked a third of the transcripts while it lasted, which shows here as a step in the share of empty text.

    python scripts/empty_text_audit.py --work $MELATI_DATA/podcast/work
"""
import argparse
import collections
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--worst", type=int, default=8)
    a = ap.parse_args()
    hours = collections.defaultdict(lambda: collections.Counter())
    per_rec = []
    for p in a.work.glob("*/fill.jsonl"):
        asr = p.parent / "asr.jsonl"
        if not asr.exists():
            continue
        h = time.strftime("%m-%d %H:00", time.localtime(asr.stat().st_mtime))
        segs = [json.loads(line) for line in asr.read_text().splitlines()]
        empty = sum(not (s.get("text") or "").strip() for s in segs)
        gaps = [g for line in p.read_text().splitlines() for g in json.loads(line).get("gaps") or []]
        c = hours[h]
        c["recordings"] += 1
        c["segments"] += len(segs)
        c["empty"] += empty
        c["gaps"] += len(gaps)
        c["gaps_empty"] += sum(g.get("kind") == "empty" for g in gaps)
        c["gaps_error"] += sum(g.get("kind") == "error" for g in gaps)
        per_rec.append((empty / max(1, len(segs)), len(segs), p.parent.name, h))
    print("hour (asr done)  recordings  segments  empty text   gaps  gaps empty  gaps error")
    for h in sorted(hours):
        c = hours[h]
        print(f"{h}  {c['recordings']:10d}  {c['segments']:8d}  {100 * c['empty'] / max(1, c['segments']):9.2f}%  "
              f"{c['gaps']:6d}  {100 * c['gaps_empty'] / max(1, c['gaps']):9.1f}%  {c['gaps_error']:10d}")
    print(f"most empty recordings (of {len(per_rec)}):")
    for share, n, rid, h in sorted(per_rec, reverse=True)[:a.worst]:
        print(f"  {rid}  {100 * share:5.1f}% of {n} segments empty  ({h})")


if __name__ == "__main__":
    main()
