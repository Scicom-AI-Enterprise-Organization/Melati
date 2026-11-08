#!/usr/bin/env python3
"""What an emotion manifest holds, in the shape of results/emotion/manifest_stats.json, and whether it holds what an
earlier one did. RUNS ON THE BOX.

Clips no encoder could decode (the <corpus>.bad.json files emotion/extract.py writes beside its features) are left
out, as emotion/probe.py leaves them out, so the counts are the ones the heads trained and were scored on. With
--expect, every corpus's split, language and per-label count must equal the earlier file's, and the script exits 1
naming each difference: heads retrained on different clips are not the same heads.

    python emotion/manifest_stats.py --manifest $MELATI_DATA/emo/manifest.jsonl \\
        --bad $MELATI_DATA/emo/feats/whisper --out $MELATI_DATA/results/emotion/manifest_stats.json \\
        [--expect $MELATI_DATA/v1/results/emotion/manifest_stats.json]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def stats(rows):
    by = {}
    for r in rows:
        c = by.setdefault(r["corpus"], {"n": 0, "lang": r["lang"], "split": r["split"], "labels": Counter()})
        c["n"] += 1
        c["labels"][r["label"]] += 1
    train = [r for r in rows if r["split"] == "train"]
    return {"clips": len(rows),
            "train": {"clips": len(train), "corpora": len({r["corpus"] for r in train}),
                      "languages": sorted({r["lang"] for r in train})},
            "held_out": {"clips": len(rows) - len(train),
                         "corpora": sorted({r["corpus"] for r in rows if r["split"] != "train"})},
            "per_label": dict(sorted(Counter(r["label"] for r in rows).items())),
            "per_corpus": {k: v | {"labels": dict(v["labels"])} for k, v in sorted(by.items())}}


def differences(new, old):
    out = []
    for c in sorted(set(new["per_corpus"]) | set(old["per_corpus"])):
        a, b = new["per_corpus"].get(c), old["per_corpus"].get(c)
        if a is None or b is None:
            out.append(f"{c}: {'missing' if a is None else 'not in the earlier manifest'}")
            continue
        for k in ("split", "lang"):
            if a[k] != b[k]:
                out.append(f"{c}: {k} {a[k]} (was {b[k]})")
        for lab in sorted(set(a["labels"]) | set(b["labels"])):
            if a["labels"].get(lab, 0) != b["labels"].get(lab, 0):
                out.append(f"{c}: {lab} {a['labels'].get(lab, 0)} (was {b['labels'].get(lab, 0)})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--bad", type=Path, default=None, help="an encoder's feature folder, for its <corpus>.bad.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect", type=Path, default=None)
    a = ap.parse_args()
    rows = [json.loads(line) for line in a.manifest.open(encoding="utf-8")]
    bad = set()
    for f in sorted(a.bad.glob("*.bad.json")) if a.bad else ():
        bad |= set(json.loads(f.read_text()))
    s = stats([r for r in rows if r["id"] not in bad]) | {"undecodable_dropped": len(bad)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(s, indent=1))
    print(f"{s['clips']} clips ({s['train']['clips']} train in {s['train']['corpora']} corpora, "
          f"{s['held_out']['clips']} held out), {len(bad)} undecodable -> {a.out}", flush=True)
    if a.expect:
        diff = differences(s, json.loads(a.expect.read_text()))
        for d in diff:
            print(f"  DIFFERS {d}", flush=True)
        print("MANIFEST_SAME" if not diff else f"MANIFEST_DIFFERS: {len(diff)}", flush=True)
        return 1 if diff else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
