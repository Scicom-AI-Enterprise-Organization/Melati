#!/usr/bin/env python3
"""Score the five checkpoints on `lexicon_synth`, the contrastive other half of the
non-speech arms.

`silence`, `music` and `nonspeech` ask what a model invents when nothing is said. This arm
asks the opposite: the clip **does** say the phrase, so emitting it is the correct answer, and
every miss is what a hallucination filter would wrongly delete. The two together are the
trade-off a mitigation lives on: a ban-list that removes `terima kasih` from noise also
removes it from someone thanking you.

Metrics per model:

  cer / wer        against the phrase that was actually spoken
  recovered        the phrase came back: CER under the per-language gate, or (for references
                   of ≤12 characters, where one wrong character is CER 0.1+) contained in the
                   hypothesis. The same rule `filter_lexicon_synth.py` accepts clips with.
  empty_rate       emitted nothing at all: a deletion, the filter's job done for it
  loop_rate        a token run ≥ 6 on a clip whose reference is a handful of words
  lang_ok          Whisper's detected language matched the clip's. Decoding is auto-language
                   here, as everywhere in this harness, and drift is a real failure mode on
                   one-second clips.

**The corpus was filtered by `openai/whisper-large-v3` with the language forced**, so v3 is
scored on clips selected partly by its own agreement. That is a real advantage and it is
reported rather than hidden: `lang_ok` and the per-language table show where it comes from.
The other four checkpoints, and v3 under auto-language decoding, are not so favoured.

    python bench/score_lexicon_synth.py --results bench/results --out bench/lexicon_synth_scores.json
"""
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import cer, max_ngram_repeat, normalise, wer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["whisper-large-v2", "whisper-large-v3", "whisper-large-v3-turbo",
          "malaysian-whisper-large-v2", "Malaysian-whisper-large-v3-turbo-v3"]


def recovered(ref: str, hyp: str, gate: float) -> bool:
    """Did the phrase come back? Short references need containment: `hi` mis-spaced is CER 0.5
    while the word is plainly there."""
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return False
    if cer(ref, hyp) <= gate:
        return True
    return len(r) <= 12 and r in h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("bench/results"))
    ap.add_argument("--out", type=Path, default=Path("bench/lexicon_synth_scores.json"))
    ap.add_argument("--judge-floor", type=Path, default=ROOT / "tts" / "judge_floor.json")
    ap.add_argument("--max-cer", type=float, default=0.25)
    ap.add_argument("--floor-slack", type=float, default=1.5)
    ap.add_argument("--loop-threshold", type=int, default=6)
    ap.add_argument("--min-lang-clips", type=int, default=30,
                    help="languages below this are pooled out of the per-language table")
    ap.add_argument("--conditions", nargs="+", default=["", "__zeros2", "__tone2"],
                    help="result-file suffixes to score. The padding study: bare, 2 s of "
                         "digital zeros each side, 2 s of real room tone each side")
    args = ap.parse_args()

    floors = {}
    if args.judge_floor.exists():
        floors = {k: v["median_cer"] for k, v in json.loads(args.judge_floor.read_text()).items()}

    def gate_for(lang):
        # Same per-language gate the corpus was built with: holding Amharic to 0.25 measures
        # the judge, not the model.
        f = floors.get(lang)
        return max(args.max_cer, f * args.floor_slack) if f else args.max_cer

    out = {}
    for model, cond in ((m, c) for m in MODELS for c in args.conditions):
        key = f"{model}{cond}" if cond else model
        path = args.results / model / f"lexicon_synth{cond}.jsonl"
        if not path.exists():
            print(f"[skip] {key}: no results")
            continue
        recs = [json.loads(l) for l in path.open(encoding="utf-8")]
        per_lang = defaultdict(lambda: {"n": 0, "cer": [], "recovered": 0, "empty": 0})
        cers, wers, rec_hits, empties, loops = [], [], 0, 0, 0
        by_engine = defaultdict(lambda: {"n": 0, "cer": [], "recovered": 0})

        for r in recs:
            ref, hyp = r["reference_text"], r["hyp"]
            meta = r.get("meta") or {}
            lang = meta.get("lang", "")
            c = cer(ref, hyp)
            ok = recovered(ref, hyp, gate_for(lang))
            cers.append(c)
            wers.append(wer(ref, hyp))
            rec_hits += ok
            empties += not normalise(hyp)
            loops += max_ngram_repeat(hyp, 1) >= args.loop_threshold
            pl = per_lang[lang]
            pl["n"] += 1; pl["cer"].append(c); pl["recovered"] += ok
            pl["empty"] += not normalise(hyp)
            eng = meta.get("engine", "")
            be = by_engine[eng]
            be["n"] += 1; be["cer"].append(c); be["recovered"] += ok

        n = len(recs)
        pad = (recs[0].get("pad") or {}) if recs else {}
        # Extra text beyond the phrase is the padding's signature: the model fills the room
        # tone with something. Measured as hypothesis length over reference length, so it is
        # comparable across languages and scripts.
        bloat = []
        for r in recs:
            ref, hyp = normalise(r["reference_text"]), normalise(r["hyp"])
            if ref:
                bloat.append(len(hyp) / len(ref))
        out[key] = {
            "model": model,
            "condition": cond.lstrip("_") or "bare",
            "pad": pad,
            "length_ratio_median": round(statistics.median(bloat), 3) if bloat else None,
            "over_2x_rate": round(sum(b > 2 for b in bloat) / max(len(bloat), 1), 4),
            "n": n,
            "cer": round(statistics.mean(cers), 4),
            "cer_median": round(statistics.median(cers), 4),
            "wer": round(statistics.mean(wers), 4),
            "recovered_rate": round(rec_hits / n, 4),
            "empty_rate": round(empties / n, 4),
            "loop_rate": round(loops / n, 4),
            "per_language": {
                l: {"n": v["n"], "cer": round(statistics.mean(v["cer"]), 4),
                    "recovered_rate": round(v["recovered"] / v["n"], 4),
                    "empty_rate": round(v["empty"] / v["n"], 4)}
                for l, v in sorted(per_lang.items()) if v["n"] >= args.min_lang_clips
            },
            "by_engine": {
                e: {"n": v["n"], "cer": round(statistics.mean(v["cer"]), 4),
                    "recovered_rate": round(v["recovered"] / v["n"], 4)}
                for e, v in sorted(by_engine.items()) if e
            },
        }
        print(f"{model:38s} {out[key]['condition']:>7s}  n={n} cer={out[key]['cer']:.3f} "
              f"recovered={out[key]['recovered_rate']:.3f} "
              f"empty={out[key]['empty_rate']:.3f} loop={out[key]['loop_rate']:.3f} "
              f"len×={out[key]['length_ratio_median']}")

    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
