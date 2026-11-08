#!/usr/bin/env python3
"""A finished run re-tagged by Melati's non-verbal heads, everything else as it ran. RUNS ON THE BOX.

A run that asked Qwen3-Omni about every gap candidate kept each answer in tags.jsonl. Here a copy of its work
directory gets the heads' answers instead: same candidates, same audio (the vocal stem), same cut and padding.
Every other stage's output is linked, not copied or recomputed, and the export stage runs again with the heads'
thresholds, so before and after differ only in who labelled the candidates (or, with --rule, also in the
rule that found them: each segment's candidates are found again from its vocal stem and its aligned words). Then:

  --score-pf OUT   score it against PodcastFillers' hand labels (scripts/eval_fillers.py), for its test episodes
  --assemble       write the export's manifest (melati.run --assemble), for scripts/compare_emilia.py
  --frames         also run the frame-level detector (melati.nv_frames) over every segment, and export its laughs
                   and coughs at --frame-thresholds where the gap labeller has none of the kind

    python nonverbal/relabel.py --work $MELATI_DATA/bench/pf_test/work_rel \\
        --sources $MELATI_DATA/bench/pf_test/sources.jsonl --models whisper --variant whisper \\
        --thresholds $MELATI_DATA/results/nonverbal/nonverbaltts_gaps_heads.json \\
        --new-work $MELATI_DATA/bench/pf_test/work_nvwhisper \\
        --score-pf $MELATI_DATA/results/fillers/podcastfillers_nvwhisper.json
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from melati import pipeline as P  # noqa: E402
from melati import nv_gaps as NG  # noqa: E402
from melati.nv_gaps import HEAD_LABELS, PAD_S, SR, clip_of  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

OWN = ("tags.jsonl", "export.jsonl")          # written here; everything else is the original run's


def link_work(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        t = dst / f.name
        if f.name in OWN or t.exists() or t.is_symlink():
            continue
        os.symlink(f, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["whisper", "wavlm", "ast"])
    ap.add_argument("--thresholds", type=Path, required=True, help="nonverbal/eval_gaps.py's output")
    ap.add_argument("--variant", default=None, help="which of its variants (eval_gaps.py) or rules (gap_rule.py "
                    "eval); default the models averaged")
    ap.add_argument("--classes", nargs="*", default=None, help="tag only these classes (those the test split measures)")
    ap.add_argument("--new-work", type=Path, required=True)
    ap.add_argument("--export-out", type=Path, default=None, help="where export writes audio (scratch)")
    ap.add_argument("--csv", type=Path, default=Path(f"{MELATI_DATA}/bench/podcastfillers/PodcastFillers.csv"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--score-pf", type=Path, default=None, help="score against PodcastFillers, write here")
    ap.add_argument("--out", type=Path, default=None, help="the older name of --score-pf")
    ap.add_argument("--assemble", action="store_true", help="also write the export's manifest")
    ap.add_argument("--rule", default=None, help="find the candidates again with melati.nv_gaps.candidates and these "
                    "keyword arguments (JSON; '{}' = the current defaults) instead of reusing the run's")
    ap.add_argument("--frames", default=None, help="the frame-level detector's encoder (whisper), from MELATI_NV_FRAMES")
    ap.add_argument("--frame-thresholds", default="{}", help='JSON, e.g. \'{"laugh": 0.995, "cough": 0.95}\'')
    args = ap.parse_args()
    args.rule = json.loads(args.rule) if args.rule is not None else None
    ev = json.loads(args.thresholds.read_text())
    variant = args.variant or "+".join(args.models)
    block = ev["variants"][variant] if "variants" in ev else ev[variant]      # eval_gaps.py's, or gap_rule.py eval's
    thr = {c: v["threshold"] for c, v in block["thresholds"].items() if v and (not args.classes or c in args.classes)}
    print(f"thresholds ({variant}, from {args.thresholds.name}): {thr}", flush=True)
    recs = [json.loads(l) for l in args.sources.open()]
    recs = [r for r in recs if (args.work / r["id"] / "tags.jsonl").exists()]
    jobs, tags = [], {}
    for rec in recs:
        d = args.work / rec["id"]
        link_work(d, args.new_work / rec["id"])
        x, _ = sf.read(str(d / "vocal16k.flac"), dtype="float32")
        segs = {s["seg"]: s for s in P.read_jsonl(d / "segments.jsonl")}
        rows = P.read_jsonl(d / "tags.jsonl")
        al = P.by_seg(d / "align.jsonl") if args.rule is not None else {}
        tags[rec["id"]] = rows
        for row in rows:
            s = segs[row["seg"]]
            xs = x[int(s["start"] * SR):int(s["end"] * SR)]
            if args.rule is not None:                    # the candidates again, under the rule asked for
                words = (al.get(row["seg"]) or {}).get("words") or []
                row["nv_gaps"] = NG.candidates(xs, words, **args.rule) if words else []
            for k, c in enumerate(row.get("nv_gaps") or []):
                jobs.append((rec["id"], row["seg"], k, clip_of(xs, c, PAD_S)))
    print(f"{len(recs)} episodes, {len(jobs)} candidates", flush=True)
    from melati.ser import HeadNV
    probs = []
    for m in args.models:
        model = HeadNV(m)
        xs = [j[3] for j in jobs]
        probs.append(np.concatenate([model(xs[s:s + args.batch])[0] for s in range(0, len(xs), args.batch)]))
        del model
    p = np.mean(probs, 0)
    by = {(r, s, k): row for (r, s, k, _), row in zip(jobs, p)}
    cfg = {"nv_thresholds": {"heads": thr}}
    if args.frames:                                      # every segment's laugh and cough probabilities
        from melati.nv_frames import FrameNV
        from melati.pipeline import FRAME_CLASSES
        fm = FrameNV(args.frames)
        segs_x = []
        for rec in recs:
            x, _ = sf.read(str(args.work / rec["id"] / "vocal16k.flac"), dtype="float32")
            segs = {s["seg"]: s for s in P.read_jsonl(args.work / rec["id"] / "segments.jsonl")}
            segs_x += [(row, x[int(segs[row["seg"]]["start"] * SR):int(segs[row["seg"]]["end"] * SR)])
                       for row in tags[rec["id"]]]
        segs_x = [(row, xs) for row, xs in segs_x if len(xs) >= int(0.3 * SR)]
        for s0 in range(0, len(segs_x), 32):
            arrs, _ = fm([xs for _, xs in segs_x[s0:s0 + 32]])
            for (row, _), a in zip(segs_x[s0:s0 + 32], arrs):
                row["nv_frames"] = {"model": fm.name, "fps": 10, **{c: [round(float(v), 3) for v in
                                                                         a[:, fm.classes.index(c)]]
                                                                     for c in FRAME_CLASSES}}
        cfg["nv_frame_thresholds"] = json.loads(args.frame_thresholds)
        print(f"frames: {len(segs_x)} segments, thresholds {cfg['nv_frame_thresholds']}", flush=True)
        del fm
    out_audio = args.export_out or args.new_work.with_name(args.new_work.name + "_export")
    for rec in recs:
        rows = tags[rec["id"]]
        for row in rows:
            for k, c in enumerate(row.get("nv_gaps") or []):
                c["dist_omni"] = c.get("dist")
                c["dist"] = {lab: round(float(v), 4) for lab, v in zip(HEAD_LABELS, by[(rec["id"], row["seg"], k)])}
                c["by"] = "heads"
        d = args.new_work / rec["id"]
        tmp = d / "tags.jsonl.tmp"
        P.write_jsonl(tmp, rows)
        tmp.replace(d / "tags.jsonl")
        (d / "export.jsonl").unlink(missing_ok=True)
        P._export(args.new_work, out_audio, rec, cfg)
    print(f"re-exported {len(recs)} recordings -> {out_audio}", flush=True)
    if args.assemble:
        subprocess.run([sys.executable, "-m", "melati.run", "--assemble", "--sources", str(args.sources),
                        "--work", str(args.new_work), "--out", str(out_audio)], check=True, cwd=ROOT)
    score = args.score_pf or args.out
    if score:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_fillers.py"), "--csv", str(args.csv),
                        "--sources", str(args.sources), "--work", str(args.new_work), "--out", str(score)],
                       check=True)
        res = json.loads(score.read_text())
        res["nv_labeller"] = {"heads": args.models, "variant": variant, "thresholds": thr,
                              "thresholds_from": str(args.thresholds),
                              "frames": args.frames and cfg.get("nv_frame_thresholds")}
        score.write_text(json.dumps(res, indent=1))
    print("RELABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
