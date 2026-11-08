#!/usr/bin/env python3
"""Score the emotion models Melati uses today on the same held-out clips as the new heads. RUNS ON THE BOX.

The current tagger averages MERaLiON-SER and emotion2vec+ (export_config `emotion_models`), served by the tag
server. Here every held-out manifest row (nEMO; NonverbalTTS's VoxCeleb half) goes through it, and the scores and
thresholds are computed by emotion/probe.py's own functions, so the two are compared on identical clips and
identical arithmetic. emotion2vec+ trained on EmoBox, which contains neither held-out set. Without --endpoints the
models are loaded in this process, as the tag server loads them (melati.emotion.load), and fed the same chunks.

    python emotion/score_current.py --manifest $MELATI_DATA/emo/manifest.jsonl \\
        [--endpoints $MELATI_DATA/serve_eval/endpoints.json] --out $MELATI_DATA/results/emotion/current.json
"""
import argparse, asyncio, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract import Audio  # noqa: E402
from probe import LABELS, scores, thresholds  # noqa: E402


async def run(rows, endpoints, models, chunk=16):
    from melati.client import Client, Endpoints
    audio = Audio()
    out = {m: np.zeros((len(rows), len(LABELS)), np.float32) for m in models}
    async with Client(Endpoints(list(endpoints))) as c:
        for s in range(0, len(rows), 256):
            clips = [audio(r["src"])[: 30 * 16000] for r in rows[s:s + 256]]
            for m in models:
                parts = await asyncio.gather(*[c.emotion(m, clips[i:i + chunk]) for i in range(0, len(clips), chunk)])
                out[m][s:s + len(clips)] = np.concatenate(parts)
            print(f"  {s + len(clips)}/{len(rows)}", flush=True)
    return out


def in_process(rows, models, chunk=16):
    from melati import emotion as EMO
    audio = Audio()
    clips = [audio(r["src"])[: 30 * 16000] for r in rows]
    out = {}
    for m in models:
        model = EMO.load(m)
        out[m] = np.concatenate([model(clips[s:s + chunk])[0] for s in range(0, len(clips), chunk)])
        print(f"  {m}: {len(clips)} clips", flush=True)
        del model
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--endpoints", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=["meralion", "e2v"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "test"]
    y = np.array([LABELS.index(r["label"]) for r in rows])
    probs = asyncio.run(run(rows, args.endpoints, args.models)) if args.endpoints else in_process(rows, args.models)
    combos = {m: probs[m] for m in args.models}
    combos["+".join(args.models)] = np.mean([probs[m] for m in args.models], 0)
    res = {}
    for name, P in combos.items():
        r = {"held_out": {}}
        for c in sorted({x["corpus"] for x in rows}):
            idx = np.array([i for i, x in enumerate(rows) if x["corpus"] == c])
            r["held_out"][c] = scores(P[idx], y[idx])
        r["held_out"]["all"] = scores(P, y)
        r["thresholds"] = thresholds(P, y)
        res[name] = r
        print(name, json.dumps({k: v["uar"] for k, v in r["held_out"].items()}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    np.savez(args.out.with_suffix(".npz"), y=y, ids=np.array([r["id"] for r in rows]), **{k: v for k, v in combos.items()})


if __name__ == "__main__":
    main()
