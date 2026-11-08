#!/usr/bin/env python3
"""Are the non-verbal tags PodcastFillers cannot score real? An independent ear decides. RUNS ON THE BOX.

PodcastFillers labelled only the regions its own detector proposed, so a breath or laugh Melati tags where it has
no label at all is unscored: the scoring calls it wrong, and it may be a breath nobody was asked about. Here AST's
own AudioSet classifier (MIT/ast-finetuned-audioset-10-10-0.4593; its labels are AudioSet's, not ours, and nothing
in Melati trained it) listens to every tag, sorted three ways by scripts/eval_fillers.py's own matching:

  matched     the tag matches a hand label of its class           (right, by the labels)
  labelled    it overlaps a hand label of another class            (wrong, by the labels)
  unlabelled  it overlaps no hand label at all                     (unscored)

beside two controls cut from the same episodes: hand-labelled Breath / Laughter events, and hand-labelled Words. If
the unlabelled tags sound to AudioSet like the matched ones and unlike words, they are what they were tagged as.
AudioSet's Breathing class barely fires on a podcast breath of a third of a second; its top class for one is Gasp or
Sigh, so `breath_like` (the share whose top class is one of BREATH_LIKE) is the comparison.

    python scripts/audit_nv_tags.py --work $MELATI_DATA/bench/pf_test/work_nvwhisper \\
        --out $MELATI_DATA/results/fillers/audit_nv_tags.json
"""
import argparse, json, os, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_fillers import inside, load_preds, load_refs, match, stem  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

SR = 16000
CLASS = {"breath": "Breathing", "laughter": "Laughter"}       # our class -> the AudioSet class that should fire
BREATH_LIKE = ("Breathing", "Gasp", "Sigh", "Snort", "Wheeze", "Pant")   # what AudioSet calls a short podcast breath
WATCH = ("Breathing", "Laughter", "Speech", "Music", "Cough", "Sigh", "Sniff", "Throat clearing", "Gasp", "Silence")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path(f"{MELATI_DATA}/bench/podcastfillers/PodcastFillers.csv"))
    ap.add_argument("--sources", type=Path, default=Path(f"{MELATI_DATA}/bench/pf_test/sources.jsonl"))
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--tol-s", type=float, default=0.2)
    ap.add_argument("--controls", type=int, default=300, help="hand-labelled events per control group")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    refs_by_ep = load_refs(args.csv, "test")
    recs = [json.loads(l) for l in args.sources.open() if l.strip()]
    items = []                                     # (class, group, rec id, start, end)
    controls = defaultdict(list)
    for rec in recs:
        refs_all = refs_by_ep.get(rec.get("stem") or stem(rec["path"]))
        preds, regions = load_preds(args.work, rec["id"])
        if refs_all is None or preds is None:
            continue
        refs = [r for r in refs_all if r[0] in ("filler", "laughter", "breath")]
        rr = [r for r in refs if inside((r[1] + r[2]) / 2, regions)]
        pairs = match(preds, rr, args.tol_s)
        mp = {i for i, _ in pairs}
        for i, (pc, ps, pe) in enumerate(preds):
            if pc not in CLASS:
                continue
            if i in mp:
                g = "matched"
            elif any(min(pe, re_) - max(ps, rs) > 0 for _, rs, re_ in refs_all):
                g = "labelled"
            else:
                g = "unlabelled"
            items.append((pc, g, rec["id"], ps, pe))
        for c, s, e in refs_all:
            if c in CLASS:
                controls[f"{c}/hand-labelled"].append((c, rec["id"], s, e))
            elif c == "Words":
                controls["words/hand-labelled"].append((c, rec["id"], s, e))
    rng = random.Random(0)
    for k, v in controls.items():
        rng.shuffle(v)
        for c, rid, s, e in v[: args.controls]:
            items.append((c if c in CLASS else "words", "control", rid, s, e))

    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(name)
    model = ASTForAudioClassification.from_pretrained(name).to("cuda").eval().half()
    idx = {lab: model.config.label2id[lab] for lab in WATCH}
    audio = {}
    pad = np.zeros(int(0.3 * SR), np.float32)
    clips = []
    for c, g, rid, s, e in items:
        if rid not in audio:
            audio[rid], _ = sf.read(str(args.work / rid / "vocal16k.flac"), dtype="float32")
        x = audio[rid]
        clips.append(np.concatenate([pad, x[int(s * SR):int(max(e, s + 0.05) * SR)][: int(2.5 * SR)], pad]))
    probs = []
    with torch.inference_mode():
        for b in range(0, len(clips), 64):
            inp = fe(clips[b:b + 64], sampling_rate=SR, return_tensors="pt").input_values.to("cuda").half()
            probs.append(torch.sigmoid(model(inp).logits.float()).cpu().numpy())
    P = np.concatenate(probs)
    groups = defaultdict(list)
    for (c, g, *_), p in zip(items, P):
        groups[f"{c}/{g}"].append(p)
    out = {}
    for k, ps in sorted(groups.items()):
        ps = np.stack(ps)
        c = k.split("/")[0]
        own = CLASS.get(c)
        top = [model.config.id2label[int(j)] for j in ps.argmax(1)]
        out[k] = {"n": len(ps), **{f"mean_{lab}": round(float(ps[:, idx[lab]].mean()), 4) for lab in WATCH}}
        if own:
            o = ps[:, idx[own]]
            out[k][f"share_{own}_above_speech"] = round(float((o > ps[:, idx["Speech"]]).mean()), 4)
            out[k][f"share_{own}_over_0.2"] = round(float((o > 0.2).mean()), 4)
        out[k]["top1"] = dict(sorted(((t, top.count(t)) for t in set(top)), key=lambda kv: -kv[1]))
        out[k]["breath_like"] = round(sum(t in BREATH_LIKE for t in top) / len(top), 4)
        print(k, json.dumps({kk: v for kk, v in out[k].items() if kk != "top1"}),
              dict(list(out[k]["top1"].items())[:6]), flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
