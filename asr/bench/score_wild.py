#!/usr/bin/env python3
"""Score the wild-audio runs against what is actually known about each clip.

`run_wild_baseline.py` writes one jsonl per model. This joins those back to the manifests and
reports the metric that fits each collection, because the collections do not share a notion of
failure:

  halas_*        real speech WITH a human-corrected reference. "Any output" is meaningless --
                 the clip contains words. What matters is error against the reference, split
                 by the human verdict, plus whether the model emits a lexicon phrase. If a
                 blocklist worked on wild audio, `lexicon_rate` would separate the flagged
                 clips from the clean ones.
  blank_speech   VAD found no speech, so the reference is the empty string and any output is
                 invented. Directly comparable to the `silence` arm.
  loop           mined because SOME model looped here; `loop_rate` says whether this one does.
  aphasia        Koenecke et al.'s confirmed triggers, no reference shipped.

Joins by id, so a model already run is never re-transcribed.

    python bench/score_wild.py --results bench/wild_results --out bench/wild_scores.json
"""
import argparse, csv, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer, normalise, wer  # noqa: E402

MODEL_ORDER = ["whisper-large-v2", "whisper-large-v3", "whisper-large-v3-turbo",
               "malaysian-whisper-large-v2", "Malaysian-whisper-large-v3-turbo-v3"]
# HALAS column stem for each checkpoint we run, where one exists.
HALAS_COL = {"whisper-large-v2": "whisper_large_v2",
             "whisper-large-v3": "whisper_large_v3",
             "whisper-large-v3-turbo": "whisper_large_v3_turbo"}


def load_manifest_index(roots):
    idx = {}
    for root in roots:
        for man in Path(root).rglob("manifest.csv"):
            collection = man.parent.parent.name
            with man.open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    r["collection"] = collection
                    idx[r.get("id", "")] = r
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, nargs="+",
                    default=[Path("bench/wild_results"), Path("bench/wild_results_mined")])
    ap.add_argument("--roots", nargs="+",
                    default=["audio_wild", "audio/aphasia_koenecke"])
    ap.add_argument("--out", type=Path, default=Path("bench/wild_scores.json"))
    ap.add_argument("--loop-threshold", type=int, default=6)
    args = ap.parse_args()

    idx = load_manifest_index(args.roots)
    print(f"[wild-score] {len(idx)} clips indexed", flush=True)

    out = {}
    for results in args.results:
        for jf in sorted(Path(results).glob("*.jsonl")):
            model = jf.stem
            recs = [json.loads(l) for l in jf.open(encoding="utf-8")]
            groups = defaultdict(list)
            for rec in recs:
                man = idx.get(rec["id"], {})
                rec["reference_text"] = man.get("reference_text", "")
                rec["_halas_label"] = man.get(f"{HALAS_COL.get(model,'')}_label", "")
                for reason in (rec.get("reasons") or "unlabelled").split("|"):
                    groups[reason].append(rec)

            entry = {}
            for reason, items in sorted(groups.items()):
                n = len(items)
                refs = [x for x in items if normalise(x["reference_text"])]
                block = {
                    "n": n,
                    "any_output_rate": round(sum(x["any_output"] for x in items) / n, 4),
                    "lexicon_rate": round(sum(x["in_lexicon"] for x in items) / n, 4),
                    "loop_rate": round(sum(x["max_token_run"] >= args.loop_threshold
                                           for x in items) / n, 4),
                }
                if refs:
                    block["n_with_reference"] = len(refs)
                    block["wer"] = round(statistics.mean(
                        wer(x["reference_text"], x["hyp"]) for x in refs), 4)
                    block["cer"] = round(statistics.mean(
                        cer(x["reference_text"], x["hyp"]) for x in refs), 4)
                    # Output longer than the reference is the shape a hallucination takes on
                    # speech: the words that were said, plus words that were not.
                    ratios = [len(normalise(x["hyp"])) / max(len(normalise(x["reference_text"])), 1)
                              for x in refs]
                    block["len_ratio_median"] = round(statistics.median(ratios), 3)
                    block["over_1_5x_rate"] = round(sum(r > 1.5 for r in ratios) / len(ratios), 4)
                block["top_outputs"] = [{"text": t[:60], "count": c} for t, c in Counter(
                    normalise(x["hyp"]) for x in items if normalise(x["hyp"])).most_common(5)]
                entry[reason] = block
            # A model appears in more than one results directory (HALAS in one, the mined
            # clips in another). Merge, or the second read silently drops the first's reasons.
            out.setdefault(model, {}).update(entry)
            print(f"{model}: " + "  ".join(f"{r}={b['n']}" for r, b in entry.items()), flush=True)

    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {args.out}")

    # Compact view of the question the arm exists to answer.
    print(f"\n{'model':<38}{'halluc CER':>11}{'clean CER':>11}{'halluc lex':>12}{'clean lex':>11}")
    for m in MODEL_ORDER:
        e = out.get(m)
        if not e or "halas_hallucination" not in e:
            continue
        h, c = e["halas_hallucination"], e.get("halas_clean", {})
        print(f"{m:<38}{h.get('cer', float('nan')):>11.3f}{c.get('cer', float('nan')):>11.3f}"
              f"{h.get('lexicon_rate', 0):>12.3f}{c.get('lexicon_rate', 0):>11.3f}")


if __name__ == "__main__":
    main()
