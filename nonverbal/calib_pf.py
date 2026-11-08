#!/usr/bin/env python3
"""Thresholds for a non-verbal head chosen on PodcastFillers' validation episodes instead of NonverbalTTS. RUNS ON
THE BOX.

The candidates are the ones nonverbal/candidates.py found in PodcastFillers' validation episodes with the pipeline's
own rule (corpus pf_cands2, features already extracted), labelled by the hand-timed event each overlaps; one that
overlaps nothing is unknown and left out, so precision here is precision where PodcastFillers has a label, the
measure the pipeline's PodcastFillers test is read by (scripts/eval_fillers.py). Per class, the lowest threshold
(melati.nv_gaps.decide's rule) whose precision reaches the target, as nonverbal/gap_rule.py chooses them on
NonverbalTTS. PodcastFillers labels no coughs, so a cough threshold must come from elsewhere (--keep copies classes
from a gap_rule.py eval file). Written in gap_rule.py eval's format, which nonverbal/relabel.py reads.

    python nonverbal/calib_pf.py --heads $MELATI_DATA/models/nv_heads_rel4 \\
        --manifest $MELATI_DATA/nv/manifest_rel4.jsonl --out .../gap_eval_rel4pf.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "emotion"))
from melati import nv_gaps as NG  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

RULE = '{"floor_db": 2}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--feats", type=Path, default=Path(f"{MELATI_DATA}/nv/feats"))
    ap.add_argument("--corpus", default="pf_cands2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--classes", nargs="+", default=["breath", "laugh"])
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--keep", type=Path, default=None, help="take the other classes' thresholds from this file")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from probe import load
    from melati.ser import head_module
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    sel = [r for r in rows if r["corpus"] == args.corpus and r["split"] == args.split]
    tmp = args.out.with_suffix(".rows.jsonl")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in sel))
    X, rs = load(tmp, args.feats, "whisper")
    tmp.unlink()
    ck = torch.load(args.heads / "whisper.pt", map_location="cpu")
    head = head_module(ck["layers"], ck["dim"], n=len(NG.HEAD_LABELS))
    head.load_state_dict(ck["state"]); head.eval()
    with torch.no_grad():
        P = torch.cat([head(torch.from_numpy(np.asarray(X[s:s + 4096])).float()).softmax(-1)
                       for s in range(0, len(X), 4096)]).numpy()
    truth = [NG.TO_CLASS.get(r["label"]) for r in rs]
    out = {}
    for cls in args.classes:
        out[cls] = None
        n_ref = sum(t == cls for t in truth)
        for t in np.arange(0.05, 0.96, 0.05):
            t = round(float(t), 2)
            hyp = [NG.decide(dict(zip(NG.HEAD_LABELS, map(float, p))), {cls: t}) == cls for p in P]
            n = sum(hyp)
            if n < 5:
                break
            right = sum(h and g == cls for h, g in zip(hyp, truth))
            if right / n >= args.target:
                out[cls] = {"threshold": t, "precision": round(right / n, 4), "recall": round(right / max(n_ref, 1), 4),
                            "hyp": n, "ref": n_ref}
                break
        print(cls, f"({n_ref} labelled in {len(rs)} candidates):", out[cls], flush=True)
    if args.keep:
        kept = json.loads(args.keep.read_text())[RULE]["thresholds"]
        out.update({c: v for c, v in kept.items() if c not in args.classes})
    args.out.write_text(json.dumps({RULE: {"thresholds": out, "chosen_on": f"{args.corpus} {args.split}",
                                           "target": args.target}}, indent=1))
    print("CALIB_PF_DONE", flush=True)


if __name__ == "__main__":
    main()
