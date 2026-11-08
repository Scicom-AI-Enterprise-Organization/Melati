#!/usr/bin/env python3
"""How much speech export drops because the transcriber wrote nothing. RUNS ON THE BOX.

Export skips a segment whose transcript is empty (melati/pipeline.py), so a transcriber that stops on speech costs
yield without any error. Same segments, different transcribers: compare work dirs that share stage C's output.

    python scripts/asr_empty_drops.py --out $MELATI_DATA/results/compare/empty_drops.json \\
        vox_ft=$MELATI_DATA/dev/vox/work_ft vox_rel=$MELATI_DATA/dev/vox/work_rel \\
        pil_ft=$MELATI_DATA/pilot/work_ft pil_rel=$MELATI_DATA/pilot/work_rel
"""
import argparse, json
from pathlib import Path


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def drops(work: Path):
    n = empty = 0
    s_all = s_empty = 0.0
    langs = {}
    for rec in sorted(p for p in work.iterdir() if p.is_dir()):
        segs = {s["seg"]: s for s in rows(rec / "segments.jsonl")}
        for a in rows(rec / "asr.jsonl"):
            s = segs.get(a["seg"])
            if s is None:
                continue
            dur = s["end"] - s["start"]
            n += 1; s_all += dur
            if not (a.get("text") or "").strip():
                empty += 1; s_empty += dur
                langs[a.get("language")] = langs.get(a.get("language"), 0) + 1
    return {"work": str(work), "segments": n, "empty": empty, "empty_share": round(empty / max(n, 1), 4),
            "hours": round(s_all / 3600, 3), "hours_empty": round(s_empty / 3600, 3), "empty_by_language": langs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=work_dir")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = {}
    for run in args.runs:
        label, work = run.split("=", 1)
        res[label] = drops(Path(work))
        print(label, json.dumps(res[label]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
