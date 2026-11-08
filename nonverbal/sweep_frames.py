#!/usr/bin/env python3
"""The frame head's laughs on PodcastFillers' test episodes over a range of thresholds, through the pipeline's own
export and scorer, to tell a threshold from a model. The release tags laughs at 0.9, chosen on the validation
episodes for 0.8 precision where labelled; the previous release and Qwen3-Omni sit at higher precision and lower
recall, and a model is only behind if no threshold of its own reaches their point. CPU only: the frame
probabilities are the ones nonverbal/relabel.py --frames kept in each segment's tags.jsonl, and every other export
setting is the one that run used. RUNS ON THE BOX.

    python nonverbal/sweep_frames.py --work $MELATI_DATA/bench/pf_test/work_nvrel4f \\
        --sources $MELATI_DATA/bench/pf_test/sources.jsonl \\
        --base $MELATI_DATA/results/fillers/podcastfillers_nvrel4f.json \\
        --thresholds 0.9 0.95 0.97 0.98 0.99 0.995 0.998 --out $MELATI_DATA/results/nonverbal/frames_sweep_pf.json
"""
import argparse
import json
import os
from collections import Counter
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from melati import pipeline as P  # noqa: E402
from nonverbal.release import lab_p  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

R = Path(f"{MELATI_DATA}/results/fillers")


def point(d, c="laughter"):
    m = d["scopes"]["inside"][c]
    return {"recall": m["recall"], "precision_labelled": lab_p(d, c), "pred": m["pred"], "ref": m["ref"],
            "recall_all": d["scopes"]["all"][c]["recall"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="a relabel.py --frames work directory")
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True, help="that run's score, for the settings it exported with")
    ap.add_argument("--thresholds", nargs="+", type=float, required=True)
    ap.add_argument("--scratch", type=Path, default=Path(f"{MELATI_DATA}/bench/pf_test/sweep_frames"))
    ap.add_argument("--csv", type=Path, default=Path(f"{MELATI_DATA}/bench/podcastfillers/PodcastFillers.csv"))
    ap.add_argument("--audioset-laugh", nargs="*", type=float, default=[],
                    help="also: the first --thresholds value with AudioSet's laughs added at each of these "
                         "(nonverbal/audioset_pass.py must have run on --work)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lab = json.loads(a.base.read_text())["nv_labeller"]
    recs = [json.loads(line) for line in a.sources.open()]
    recs = [r for r in recs if (a.work / r["id"] / "tags.jsonl").exists()]
    res = {"work": str(a.work), "exported_with": lab, "laugh": {},
           "others": {"previous release": point(json.loads((R / "podcastfillers_nvrel2.json").read_text())),
                      "Qwen3-Omni": point(json.loads((R / "podcastfillers_rel.json").read_text()))}}
    from melati.stage_export import DEFAULTS
    runs = [(str(t), t, None) for t in a.thresholds] + [(f"{a.thresholds[0]}+audioset_{u}", a.thresholds[0], u)
                                                        for u in a.audioset_laugh]
    res["tags_per_hour"] = {}
    for key, t, u in runs:
        nw = a.scratch / f"laugh_{key}"
        cfg = {"nv_thresholds": {"heads": lab["thresholds"]}, "nv_frame_thresholds": dict(lab["frames"], laugh=t),
               "audioset_thresholds": dict(DEFAULTS["audioset_thresholds"], **({"laugh": u} if u else {}))}
        kinds, secs = Counter(), 0.0
        for rec in recs:
            src, dst = a.work / rec["id"], nw / rec["id"]
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.name != "export.jsonl" and not (dst / f.name).is_symlink():
                    os.symlink(f.resolve(), dst / f.name)
            rows = [row for row, _ in P.export_rows(nw, rec, cfg, audio=False)]
            P.write_jsonl(dst / "export.jsonl", rows)
            for row in rows:
                ann = json.loads(row["annotation"])
                kinds.update(e["kind"] for e in ann.get("events") or [])
                kinds.update(f"<{d['label']}>" for d in ann.get("delivery") or [])
                secs += row["duration"]
        score = nw / "score.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_fillers.py"), "--csv", str(a.csv),
                        "--sources", str(a.sources), "--work", str(nw), "--out", str(score)],
                       check=True, stdout=subprocess.DEVNULL)
        res["laugh"][key] = p = point(json.loads(score.read_text()))
        res["tags_per_hour"][key] = {k: round(v / (secs / 3600), 2) for k, v in kinds.most_common()}
        print(f"laugh {key}: found {p['recall']:.3f}, precise where labelled {p['precision_labelled']:.3f}, "
              f"{p['pred']} tags; per hour {res['tags_per_hour'][key]}", flush=True)
    for k, p in res["others"].items():
        print(f"{k}: found {p['recall']:.3f}, precise where labelled {p['precision_labelled']:.3f}, {p['pred']} tags")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
