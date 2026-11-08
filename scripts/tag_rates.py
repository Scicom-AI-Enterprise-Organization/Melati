#!/usr/bin/env python3
"""Tags an hour of exported speech, by kind and by source, over the recordings nonverbal/audioset_pass.py has run on:
the export as it stands (melati.stage_export.DEFAULTS) against the same export without AudioSet's taggers, to say what
they add. Counts only, no text. RUNS ON THE BOX.

    python scripts/tag_rates.py --work $MELATI_DATA/podcast/work --out $MELATI_DATA/results/dataset/tag_rates.json
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WORK = None
ARMS = {"with AudioSet": {}, "without AudioSet": {"audioset_thresholds": {}, "audioset_delivery": {}}}


def one(rid):
    from melati.pipeline import export_rows
    out = {}
    for arm, cfg in ARMS.items():
        kinds, secs = Counter(), 0.0
        for row, _ in export_rows(WORK, {"id": rid}, cfg, audio=False):
            ann = json.loads(row["annotation"])
            kinds.update(f"{e['kind']} ({e.get('source', '')})" for e in ann.get("events") or [])
            kinds.update(f"<{d['label']}>" for d in ann.get("delivery") or [])
            secs += row["duration"]
        out[arm] = (kinds, secs)
    return out


def _init(work):
    global WORK
    WORK = work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    rids = sorted(p.parent.name for p in a.work.glob("*/audioset.jsonl"))
    tot = {arm: [Counter(), 0.0] for arm in ARMS}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.work,)) as ex:
        for res in ex.map(one, rids, chunksize=8):
            for arm, (k, s) in res.items():
                tot[arm][0].update(k)
                tot[arm][1] += s
    out = {"recordings": len(rids), "hours": round(tot["with AudioSet"][1] / 3600, 1),
           "per_hour": {arm: {k: round(v / (s / 3600), 2) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}
                        for arm, (c, s) in tot.items()}}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
