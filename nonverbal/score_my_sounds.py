#!/usr/bin/env python3
"""Score the dataset's non-verbal tags and fillers on Malay speech from the sounds labelling page. RUNS ANYWHERE.

Each clip's label is its annotators' majority (ties and "unclear" give none); a tagged kind's precision is the share
of its labelled clips whose majority is that kind, and the controls (untagged word gaps) show what the page hears
where nothing was tagged. Agreement is Fleiss' kappa over clips with at least two answers.

    python nonverbal/score_my_sounds.py --sample $MELATI_DATA/nv/my_sounds/sample.jsonl \\
        --labels $MELATI_DATA/nv/my_sounds/labels.jsonl --out $MELATI_DATA/results/nonverbal/my_sounds.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HEARD = {"laugh": "laughs", "breath": "breathes", "cough": "coughs", "filler (um, erm, eh)": "filler",
         "speech (a word)": "speech", "other sound": "other", "nothing": "nothing"}


def fleiss(rows, cats):
    rows = [r for r in rows if sum(r.values()) >= 2]
    if not rows:
        return None
    p_i = [(sum(v * v for v in r.values()) - sum(r.values())) / (sum(r.values()) * (sum(r.values()) - 1)) for r in rows]
    tot = sum(sum(r.values()) for r in rows)
    p_j = [sum(r.get(c, 0) for r in rows) / tot for c in cats]
    pe = sum(p * p for p in p_j)
    return round((sum(p_i) / len(p_i) - pe) / (1 - pe), 3) if pe < 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    items = {json.loads(l)["id"]: json.loads(l) for l in args.sample.open(encoding="utf-8")}
    votes = defaultdict(Counter)
    who = set()
    if args.labels.exists():
        for l in args.labels.open(encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in items and r["label"] != "unclear":
                votes[r["id"]][HEARD[r["label"]]] += 1
                who.add(r["who"])
    by_kind = defaultdict(list)
    for cid, it in items.items():
        v = votes.get(cid)
        if not v:
            continue
        top = v.most_common(2)
        maj = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None
        by_kind[it["kind"]].append(maj)
    res = {"annotators": len(who), "clips": len(items), "labelled": sum(1 for v in votes.values() if v),
           "kappa": fleiss(list(votes.values()), sorted(set(HEARD.values()))), "kinds": {}}
    for kind in sorted({it["kind"] for it in items.values()}):
        m = [x for x in by_kind.get(kind, []) if x]
        res["kinds"][kind] = {"clips": sum(1 for it in items.values() if it["kind"] == kind), "with_majority": len(m),
                              "precision": round(sum(x == kind for x in m) / len(m), 3) if m and kind != "control" else None,
                              "heard": dict(Counter(m).most_common())}
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
