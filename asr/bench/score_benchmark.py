#!/usr/bin/env python3
"""Score benchmark runs into a leaderboard.

    python score_benchmark.py --results bench/results --out bench/scores.json

Metrics, by arm type:

  hallucination_rate   non-speech arms (silence / music / nonspeech). Reference is the
                       empty string, so ANY output is a hallucination. Reported raw --
                       published figures disagree by ~10x purely over whether Whisper's
                       own no-speech filter is applied first, so state which you mean.
  loop_rate/max_run    longest CONSECUTIVE n-gram run. Consecutive runs are the loop
                       signal; duplicates-anywhere also fire on legitimately repetitive
                       speech.
  overgen_p50/p95      reduplication only: emitted repeats / true repeats. 1.0 correct,
                       >1 runaway, <1 a mitigation that erased genuine repetition.
  wer/cer              arms with a real transcript -- the regression guard.
  lexicon_rate         output is exactly a known hallucination phrase.
  lang_drift           share whose detected language differs from the arm's expected one.
"""
import argparse, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (cer, in_lexicon, load_lexicon, max_ngram_repeat,  # noqa: E402
                     normalise, overgeneration_ratio, wer)

BLANK_ARMS = {"silence", "music", "nonspeech"}


def score_file(path: Path, lexicon, loop_n, loop_thresh) -> dict:
    recs = [json.loads(l) for l in path.open(encoding="utf-8")]
    if not recs:
        return {}
    arm, n = path.stem, len(recs)
    runs = [max_ngram_repeat(r["hyp"], loop_n) for r in recs]
    out = {
        "arm": arm, "n": n,
        "loop_rate": round(sum(x >= loop_thresh for x in runs) / n, 4),
        "max_run": max(runs),
        "lexicon_rate": round(sum(in_lexicon(r["hyp"], lexicon) for r in recs) / n, 4),
        "empty_rate": round(sum(not normalise(r["hyp"]) for r in recs) / n, 4),
    }
    if arm in BLANK_ARMS:
        # Two defensible readings, and they differ a lot -- Whisper emits bare "." on silence
        # very often, which is output but not fabricated *content*. Report both rather than
        # pick one silently; this is the same ambiguity that makes published figures disagree.
        out["hallucination_rate_any_output"] = round(
            sum(bool(r["hyp"].strip()) for r in recs) / n, 4)          # any characters at all
        out["hallucination_rate"] = round(
            sum(bool(normalise(r["hyp"])) for r in recs) / n, 4)        # actual word content
        top = Counter(r["hyp"].strip() for r in recs if r["hyp"].strip()).most_common(5)
        out["top_hallucinations"] = [{"text": t[:80], "count": c} for t, c in top]
    refs = [r for r in recs if r.get("reference_text")]
    if refs and arm != "reduplication":
        out["wer"] = round(statistics.mean(wer(r["reference_text"], r["hyp"]) for r in refs), 4)
        out["cer"] = round(statistics.mean(cer(r["reference_text"], r["hyp"]) for r in refs), 4)
    if arm == "fleurs":
        # Micro wer/cer above would let a 20-clip language with clean audio offset one that
        # collapsed, and the point of the arm is to catch exactly that. Average per LANGUAGE.
        # CER is the headline: zh/ja/th/lo/my/km have no word spaces.
        per = defaultdict(lambda: {"wer": [], "cer": []})
        for r in recs:
            if not (r.get("reference_text") or "").strip():
                continue
            lang = (r.get("meta") or {}).get("lang") or "??"
            per[lang]["wer"].append(wer(r["reference_text"], r["hyp"]))
            per[lang]["cer"].append(cer(r["reference_text"], r["hyp"]))
        if per:
            out["n_langs"] = len(per)
            out["cer_macro"] = round(statistics.mean(
                statistics.mean(v["cer"]) for v in per.values()), 4)
            out["wer_macro"] = round(statistics.mean(
                statistics.mean(v["wer"]) for v in per.values()), 4)
            # The macro mean is dominated by the languages Whisper simply cannot transcribe
            # (am 1.81, km 1.58, so 1.44 for large-v3). The median over languages says what
            # the typical language looks like; report both or the number is misread.
            out["cer_median_lang"] = round(statistics.median(
                statistics.mean(v["cer"]) for v in per.values()), 4)
            out["per_lang_cer"] = {k: round(statistics.mean(v["cer"]), 4)
                                   for k, v in sorted(per.items())}
    if arm == "reduplication":
        ratios = sorted(overgeneration_ratio(r["hyp"], r["meta"].get("unit", ""),
                                             int(r["meta"].get("n_repeats", 0)))
                        for r in recs if r["meta"].get("unit"))
        if ratios:
            out["overgen_p50"] = round(statistics.median(ratios), 3)
            out["overgen_p95"] = round(ratios[min(len(ratios) - 1, int(0.95 * len(ratios)))], 3)
            out["overgen_max"] = round(ratios[-1], 3)
            out["runaway_rate"] = round(sum(x > 1.5 for x in ratios) / len(ratios), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("bench/results"))
    ap.add_argument("--lexicon", type=Path, default=Path("combined_lexicon.csv"))
    ap.add_argument("--out", type=Path, default=Path("bench/scores.json"))
    ap.add_argument("--loop-n", type=int, default=1)
    ap.add_argument("--loop-threshold", type=int, default=6)
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon) if args.lexicon.exists() else set()
    if not lex:
        print(f"[warn] no lexicon at {args.lexicon}; lexicon_rate will be 0")

    scores = {}
    for mdir in sorted(p for p in args.results.glob("*") if p.is_dir()):
        per_arm = {}
        for jf in sorted(mdir.glob("*.jsonl")):
            s = score_file(jf, lex, args.loop_n, args.loop_threshold)
            if s:
                per_arm[s.pop("arm")] = s
        if per_arm:
            scores[mdir.name] = per_arm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2, ensure_ascii=False))

    hdr = f"{'model':<26}{'arm':<18}{'n':>6}{'halluc':>9}{'any':>8}{'loop':>8}{'maxrun':>8}{'ovg95':>8}{'wer':>8}"
    print(hdr); print("-" * len(hdr))
    for m, arms in scores.items():
        for a, s in arms.items():
            def col(k, spec=">8.3f"):
                v = s.get(k)
                return format(v, spec) if isinstance(v, float) else format("-", ">8")
            hr = s.get("hallucination_rate")
            hr_s = format(f"{hr:.1%}", ">9") if isinstance(hr, float) else format("-", ">9")
            ha = s.get("hallucination_rate_any_output")
            ha_s = format(f"{ha:.1%}", ">8") if isinstance(ha, float) else format("-", ">8")
            print(f"{m:<26}{a:<18}{s['n']:>6}{hr_s}{ha_s}"
                  f"{col('loop_rate')}{s.get('max_run', '-'):>8}"
                  f"{col('overgen_p95')}{col('wer')}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
