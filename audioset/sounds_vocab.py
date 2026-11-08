#!/usr/bin/env python3
"""Which of AudioSet's sounds Melati writes into a transcript as point tags ([music], [applause], [dog]), and at what
threshold. audioset/timing_eval.py measured what places a sound in time best: the average of PretrainedSED's frame
taggers and the frame head trained beside v3 (frame mAP 0.4410 on AudioSet's strongly labelled test clips, against
0.4261 for PretrainedSED alone and 0.2970 for v4 read off windows). For each class both frame models have, the lowest
threshold at which that average's 160 ms frames are PRECISION right on those clips is kept only where it rests on
SUPPORT clips and still finds MIN_RECALL of the class's frames, and never under FLOOR. A threshold is set where the
class is as common as it is in AudioSet's test clips (a third of them are music), and a podcast is not that: where a
sound is rarer the same threshold lets a larger share of wrong frames through, hence the stricter precision and the
floor, and the gate on v4 the export applies (each class's `gate`; STRICT raises both for a sound that sits under
speech). Left out by rule: the human voice
(speech, and the sounds Melati's own non-verbal tags cover), the respiratory sounds, the channel and environment
classes (what a recording sounds like, not something in it), every music class but Music itself, and a few too
generic to read as a tag. RUNS ON THE BOX, in .venv-sed; melati/sounds_inline.json is a copy of what it writes.

    python audioset/sounds_vocab.py --out $MELATI_DATA/results/audioset/sounds_inline.json
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import ROOT  # noqa: E402
import timing_eval as T  # noqa: E402

TIMING = "mean of the two frame models"
PRECISION, SUPPORT, MIN_RECALL, FLOOR, GATE = 0.9, 20, 0.05, 0.5, 0.5
# A sound that sits under speech gets a stricter floor and gate. In the 5-hour pilot's listening check most [music] tags
# at 0.5 were faint background music under the voice, so only confident ones are kept: at 0.8 a tag needs both v4 and
# the frame models confident. At 0.8 on every class only 22 of the 81 classes keep a threshold, so the other classes
# keep the measured 0.5 floor.
STRICT = {"Music": 0.8}
UNDER = ("Human voice", "Respiratory sounds", "Channel, environment and background")
ONLY_ITSELF = ("Music",)                     # of its subtree, only the class itself
GENERIC = ("Silence", "Sound effect", "Noise", "Inside, small room", "Outside, urban or manmade")


def pick(job):
    """One class's lowest threshold, at its floor or above, at which its frames are PRECISION right, with the precision
    and recall there (precision need not rise with the threshold, so the floor is part of the search)."""
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    c, floor = job
    p, r, t = precision_recall_curve(T.S["fy"][:, :, c].ravel(), T.timing()[TIMING](c))
    ok = np.flatnonzero((p[:-1] >= PRECISION) & (t >= floor))
    if not len(ok):
        return None
    k = ok[0]
    return {"threshold": round(float(t[k]), 4), "precision": round(float(p[k]), 4), "recall": round(float(r[k]), 4)}


def tag_of(name):
    """AudioSet's display name as a point tag: its first comma-separated part, in lower case ("Beep, bleep" -> beep)."""
    return name.split(",")[0].strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    names, shared, _ = T.arrays()
    fy = T.S["fy"]
    nodes = json.loads((ROOT / "ontology.json").read_text())
    by_id, by_name = {n["id"]: n for n in nodes}, {n["name"]: n for n in nodes}
    parents = {}
    for n in nodes:
        for c in n["child_ids"]:
            parents.setdefault(c, []).append(n["id"])

    def above(mid):
        seen, todo = set(), list(parents.get(mid, []))
        while todo:
            m = todo.pop()
            if m not in seen:
                seen.add(m)
                todo += parents.get(m, [])
        return seen
    out = {name: set() for name in UNDER + ONLY_ITSELF}
    for name in out:
        todo = [by_name[name]["id"]]
        while todo:
            m = todo.pop()
            out[name].add(m)
            todo += by_id[m]["child_ids"]
    left_out, cands = {}, []
    for c in shared:
        mid = by_name[names[c]]["id"]
        why = next((f"under {u}" for u in UNDER if mid in out[u]), None)
        why = why or next((f"under {u}, not {u} itself" for u in ONLY_ITSELF if mid in out[u] and names[c] != u), None)
        why = why or ("too generic" if names[c] in GENERIC else None)
        if why:
            left_out.setdefault(why, []).append(names[c])
        else:
            cands.append(c)
    with Pool(a.workers) as pool:
        th = pool.map(pick, [(c, STRICT.get(names[c], FLOOR)) for c in cands])
    keep = {}
    for c, t in zip(cands, th):
        clips = int(fy[:, :, c].any(1).sum())
        why = (f"never {PRECISION} precise at its floor or above" if t is None else f"under {SUPPORT} test clips"
               if clips < SUPPORT else f"recall under {MIN_RECALL}" if t["recall"] < MIN_RECALL else None)
        if why:
            left_out.setdefault(why, []).append(names[c])
            continue
        keep[names[c]] = dict(t, gate=max(GATE, STRICT.get(names[c], 0.0)), floor=STRICT.get(names[c], FLOOR),
                              tag=tag_of(names[c]), positive_clips=clips, mid=by_name[names[c]]["id"])
    mids = {v["mid"]: k for k, v in keep.items()}
    for k, v in keep.items():
        v["ancestors"] = sorted(mids[m] for m in above(v["mid"]) if m in mids)
    tags = [v["tag"] for v in keep.values()]
    assert len(set(tags)) == len(tags), "two classes would write the same tag"
    res = {"timing": "the mean of PretrainedSED's BEATs+ATST-F+M2D and the frame head trained beside v3 "
                     "(audioset/frames_train.py, $MELATI_DATA/models/audioset/frames1/best.pt)",
           "step_s": 0.16, "precision": PRECISION, "support_clips": SUPPORT, "min_recall": MIN_RECALL, "floor": FLOOR,
           "gate": GATE, "strict": STRICT,
           "classes": dict(sorted(keep.items(), key=lambda kv: -kv[1]["recall"])),
           "left_out": {k: sorted(v) for k, v in left_out.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[vocab] {len(keep)} tags of {len(shared)} classes both frame models have -> {a.out}", flush=True)
    for why, v in res["left_out"].items():
        print(f"  left out, {why}: {len(v)}", flush=True)
    for k, v in res["classes"].items():
        print(f"  [{v['tag']}] threshold {v['threshold']:.3f} (v4 at {v['gate']}), precision {v['precision']:.3f}, recall "
              f"{v['recall']:.3f}, {v['positive_clips']} clips"
              f"{', under ' + ', '.join(v['ancestors']) if v['ancestors'] else ''}", flush=True)


if __name__ == "__main__":
    main()
