#!/usr/bin/env python3
"""Unlabelled podcast clips to distil an emotion head on, from Melati's published shards, with the strong teacher's
probabilities for each. RUNS ON THE BOX.

A clip is a segment of 2 to 12 s whose language is Malay or English (the Tamil and Mandarin rows are the ASR
language fallback's mistakes), at most --per-rec from one recording so no podcast outweighs the rest, drawn at random.
Its src points into its shard's audio column, which emotion/extract.py reads as it is; its corpus is
podcast_<the --out folder's name>, so a trial pool's features never stand in for the real one's. The strong teacher is the seven
models the pipeline already averages for every segment (melati/stage_export.py emotion_models: Melati's five heads,
trained on all 21 corpora, MERaLiON-SER and emotion2vec+), whose segment probabilities are in the run's tags.jsonl.

    python emotion/pool.py --hf $MELATI_DATA/podcast/hf --work $MELATI_DATA/podcast/work --n 30000 \\
        --out $MELATI_DATA/emo/pool
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati.ser import LABELS  # noqa: E402

STRONG = ("h_whisper", "h_whisper_ft", "h_w2vbert", "h_wavlm", "h_e2v", "meralion", "e2v")


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--per-rec", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cand = defaultdict(list)                              # recording -> [(shard, row, id, language)]
    for shard in sorted((a.hf / "data").glob("train-*.parquet")):
        t = pq.read_table(shard, columns=["id", "recording", "duration", "language"]).to_pylist()
        for i, r in enumerate(t):
            if 2.0 <= r["duration"] <= 12.0 and r["language"] in ("ms", "en"):
                cand[r["recording"]].append((str(shard), i, r["id"], r["language"]))
    rnd = random.Random(a.seed)
    picked = []
    for rec in sorted(cand):
        rows = cand[rec]
        picked += rnd.sample(rows, min(a.per_rec, len(rows)))
    rnd.shuffle(picked)
    picked = picked[:a.n]
    by_rec = defaultdict(set)
    for _, _, sid, _ in picked:
        by_rec[sid.rsplit("_", 1)[0]].add(sid)
    teacher = {}
    for rec, ids in by_rec.items():
        for line in (a.work / rec / "tags.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["seg"] in ids:
                ps = [m["segment"] for k, m in (r.get("emotion") or {}).items() if k in STRONG and m.get("segment")]
                if len(ps) == len(STRONG):
                    teacher[r["seg"]] = np.mean(ps, 0)
    keep = sorted((p for p in picked if p[2] in teacher), key=lambda p: (p[0], p[1]))   # shard by shard, for extract.py
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for shard, row, sid, lang in keep:
            f.write(json.dumps({"id": sid, "corpus": f"podcast_{a.out.name}", "split": "pool", "label": None, "lang": lang,
                                "src": {"parquet": shard, "col": "audio", "row": row}}) + "\n")
    P = np.stack([teacher[p[2]] for p in keep]).astype(np.float32)
    P /= P.sum(1, keepdims=True)
    np.savez(a.out / "teacher_strong.npz", ids=np.array([p[2] for p in keep]), p=P)
    print(json.dumps({"clips": len(keep), "recordings": len({p[2].rsplit('_', 1)[0] for p in keep}),
                      "languages": {l: sum(p[3] == l for p in keep) for l in ("ms", "en")},
                      "strong_teacher_argmax": {LABELS[c]: int((P.argmax(1) == c).sum()) for c in range(len(LABELS))}}))


if __name__ == "__main__":
    main()
