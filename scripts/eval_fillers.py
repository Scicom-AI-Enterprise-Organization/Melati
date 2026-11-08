#!/usr/bin/env python3
"""Fillers, laughter and breaths in Melati's output against PodcastFillers' hand labels. RUNS ON THE BOX.

PodcastFillers (Zhu, Caceres and Salamon, Interspeech 2022) labels 85,803 events in 199 English
podcast episodes, each timed in its episode: Uh and Um (the fillers, ~35k), Laughter, Breath, and
words, repetitions, music, agreement and noise. Its episodes are split by show, so the test split
is speakers nobody trained on. Melati is run on the test episodes like on any podcast
(deploy/box_pilot.sh), and this compares what it exported with the labels:

  predicted  filler   every annotation word flagged `filler` (a filler Whisper wrote, or a voiced
                      gap that transcribed alone as one), at its aligned time
             laughter / breath   annotation events `laughs` / `breathes`
  reference  label_consolidated_vocab Uh or Um -> filler; Laughter; Breath (test split: 2,598 Uh,
             2,446 Um, 579 Laughter, 732 Breath, in 20 episodes)

An event matches a reference of its class when the two intervals overlap, or their centres are
within --tol-s of each other; matching is one to one, best overlap first. Two scopes:

  inside   references whose centre lies in a segment Melati exported: how well it TAGS what it keeps
  all      every reference in the episode: tagging times yield, the number a user of the data meets

Also reported: what the unmatched predictions are instead (the reference class they overlap, e.g.
a "filler" that was a Word or a Breath), which is where a false positive comes from.

    python scripts/eval_fillers.py --csv $MELATI_DATA/bench/podcastfillers/PodcastFillers.csv \\
        --sources $MELATI_DATA/bench/podcastfillers/sources_test.jsonl \\
        --work $MELATI_DATA/bench/podcastfillers/work --out results/fillers/podcastfillers.json
"""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

REF = {"Uh": "filler", "Um": "filler", "Laughter": "laughter", "Breath": "breath"}
PRED_EVENT = {"laughs": "laughter", "breathes": "breath"}


def stem(name: str) -> str:
    """The episode name without its audio extension. Not Path.stem: PodcastFillers' names carry no
    extension and some have a dot in the title ("Ep. 54 ..."), which Path.stem would cut at."""
    for ext in (".mp3", ".wav", ".flac", ".m4a"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def load_refs(path: Path, split: str):
    """{episode stem: [(class or other label, start, end)]} for episodes of `split`."""
    out = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r.get("episode_split_subset") or "").strip() != split:
                continue
            lab = (r.get("label_consolidated_vocab") or r.get("consolidated_label") or "").strip()
            try:
                s, e = float(r["event_start_inepisode"]), float(r["event_end_inepisode"])
            except (KeyError, ValueError):
                continue
            out[stem(r["podcast_filename"])].append((REF.get(lab, lab or "None"), s, e))
    return out


def load_preds(work: Path, rec_id: str):
    """-> (predicted events [(class, start, end)], exported regions [(start, end)]) in recording time."""
    f = work / rec_id / "export.jsonl"
    if not f.exists():
        return None, None
    preds, regions = [], []
    for line in f.open(encoding="utf-8"):
        row = json.loads(line)
        t0 = float(row["start"])
        regions.append((t0, float(row["end"])))
        ann = json.loads(row["annotation"]) if isinstance(row.get("annotation"), str) else row.get("annotation") or {}
        for w in ann.get("words") or []:
            if w.get("filler"):
                preds.append(("filler", t0 + w["start"], t0 + w["end"]))
        for ev in ann.get("events") or []:
            cls = PRED_EVENT.get(ev.get("kind"))
            if cls:
                preds.append((cls, t0 + ev["start"], t0 + ev["end"]))
    return preds, regions


def match(preds, refs, tol):
    """One-to-one matching of same-class intervals, best overlap first. -> set of (pred i, ref j)."""
    cands = []
    for i, (pc, ps, pe) in enumerate(preds):
        for j, (rc, rs, re_) in enumerate(refs):
            if pc != rc:
                continue
            ov = min(pe, re_) - max(ps, rs)
            if ov > 0 or abs((ps + pe) / 2 - (rs + re_) / 2) <= tol:
                cands.append((ov, i, j))
    used_p, used_r, pairs = set(), set(), set()
    for ov, i, j in sorted(cands, reverse=True):
        if i not in used_p and j not in used_r:
            used_p.add(i); used_r.add(j); pairs.add((i, j))
    return pairs


def inside(t, regions):
    return any(a <= t <= b for a, b in regions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--sources", type=Path, required=True, help="the episodes Melati processed (melati.sources)")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tol-s", type=float, default=0.2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    refs_by_ep = load_refs(args.csv, args.split)
    recs = [json.loads(l) for l in args.sources.open() if l.strip()]
    classes = ("filler", "laughter", "breath")
    st = {scope: defaultdict(Counter) for scope in ("inside", "all")}
    confusions = defaultdict(Counter)
    episodes, missing = 0, []
    for rec in recs:
        refs_all = refs_by_ep.get(rec.get("stem") or stem(rec["path"]))
        preds, regions = load_preds(args.work, rec["id"])
        if refs_all is None or preds is None:
            missing.append(rec.get("stem") or rec["id"])
            continue
        episodes += 1
        refs = [r for r in refs_all if r[0] in classes]
        for scope in ("inside", "all"):
            rr = [r for r in refs if scope == "all" or inside((r[1] + r[2]) / 2, regions)]
            pairs = match(preds, rr, args.tol_s)
            mp, mr = {i for i, _ in pairs}, {j for _, j in pairs}
            for c in classes:
                s = st[scope][c]
                s["ref"] += sum(1 for r in rr if r[0] == c)
                s["ref_found"] += sum(1 for j, r in enumerate(rr) if r[0] == c and j in mr)
                s["pred"] += sum(1 for p in preds if p[0] == c)
                s["pred_right"] += sum(1 for i, p in enumerate(preds) if p[0] == c and i in mp)
        # what an unmatched prediction overlaps instead (any labelled class, Words included)
        pairs = match(preds, refs, args.tol_s)
        mp = {i for i, _ in pairs}
        for i, (pc, ps, pe) in enumerate(preds):
            if i in mp:
                continue
            hit = [rc for rc, rs, re_ in refs_all if min(pe, re_) - max(ps, rs) > 0]
            confusions[pc][hit[0] if hit else "(no label)"] += 1
    res = {"split": args.split, "episodes": episodes, "not_found": missing[:20], "tol_s": args.tol_s,
           "scopes": {scope: {c: {"precision": round(s["pred_right"] / s["pred"], 4) if s["pred"] else None,
                                  "recall": round(s["ref_found"] / s["ref"], 4) if s["ref"] else None, **dict(s)}
                              for c, s in st[scope].items()} for scope in st},
           "unmatched_predictions_overlap": {c: dict(v.most_common(8)) for c, v in confusions.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"{episodes} episodes ({len(missing)} not found)")
    for scope in ("inside", "all"):
        for c, r in res["scopes"][scope].items():
            print(f"  {scope:6s} {c:9s} P {r['precision']}  R {r['recall']}  (pred {r['pred']}, ref {r['ref']})")
    print("unmatched predictions overlap:", json.dumps(res["unmatched_predictions_overlap"]))


if __name__ == "__main__":
    main()
