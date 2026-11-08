#!/usr/bin/env python3
"""Fold every scored ASR run into one aggregate JSON for the paper (results/asr/summary.json).

Reads <runs>/<run>/eval.json (asr/bench/score_eval_run.py) and <runs>/<run>/run.json, and keeps
the numbers the paper quotes: hallucination on the three blank arms, words over wild voice-free
audio, phrase recovery, the repetition arm, English WER, and the two multilingual guards
(FLEURS, never trained on; CV22 test, in-domain for the replay mixes), macro over languages,
with the per-language CER kept for the per-language figure.

    python asr/summarise.py --runs $MELATI_DATA/asr/runs --out results/asr/summary.json
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from metrics import normalise  # noqa: E402
from bench_protocol import modes as bench_modes  # noqa: E402


def wild_words(run: Path):
    """The paper's `wildWd`: the share of VAD-confirmed voice-free wild clips on which the model
    writes any WORD (eval.json's any_output_rate also counts a bare full stop)."""
    for f in sorted((run / "bench").rglob("wild.jsonl")):
        recs = [json.loads(l) for l in f.open(encoding="utf-8")]
        bs = [r for r in recs if "blank_speech" in ((r.get("meta") or {}).get("reasons") or "")]
        if bs:
            return round(sum(bool(normalise(r["hyp"])) for r in bs) / len(bs), 4), len(bs)
    return None, 0


SPEECH_ARMS = ("librispeech_test_clean", "genuine", "speech_in_noise", "fleurs", "cv22", "lexicon_synth")


def empty_on_speech(run: Path) -> dict:
    """Per speech arm, the share of clips WITH a reference that came back empty: a model allowed to stop
    (decoded as vLLM serves it) can stop on speech too, and every such clip counts as a full deletion."""
    out = {}
    for arm in SPEECH_ARMS:
        f = next(iter(sorted((run / "bench").glob(f"*/{arm}.jsonl"))), None)
        if f is None:
            continue
        rows = [r for r in map(json.loads, f.open(encoding="utf-8")) if normalise(r.get("reference_text") or "")]
        if rows:
            out[arm] = round(sum(not normalise(r["hyp"]) for r in rows) / len(rows), 4)
    return out


def pick(ev: dict) -> dict:
    g = lambda *k: _get(ev, *k)
    wild = ev.get("wild") or {}
    blank = wild.get("blank_speech") or {}
    return {
        "silence_hr": g("silence", "hallucination_rate"),
        "music_hr": g("music", "hallucination_rate"),
        "nonspeech_hr": g("nonspeech", "hallucination_rate"),
        "wild_any_output": blank.get("any_output_rate"),
        "wild_loop": blank.get("loop_rate"),
        "lexicon_recovered": g("lexicon_synth", "recovered_rate"),
        "redup_runaway": g("reduplication", "runaway_rate"),
        "redup_empty": g("reduplication", "empty_rate"),
        "librispeech_wer": g("librispeech_test_clean", "wer"),
        "genuine_wer": g("genuine", "wer"),
        "speech_in_noise_wer": g("speech_in_noise", "wer"),
        "librispeech_wer_corpus": g("librispeech_test_clean", "wer_corpus"),
        "genuine_wer_corpus": g("genuine", "wer_corpus"),
        "speech_in_noise_wer_corpus": g("speech_in_noise", "wer_corpus"),
        "fleurs_cer_macro": g("fleurs", "cer_macro"),
        "fleurs_cer_median_lang": g("fleurs", "cer_median_lang"),
        "fleurs_n_langs": g("fleurs", "n_langs"),
        "fleurs_per_lang": {k: v["cer"] for k, v in (g("fleurs", "per_lang") or {}).items()},
        "cv22_cer_macro": g("cv22", "cer_macro"),
        "cv22_cer_median_lang": g("cv22", "cer_median_lang"),
        "cv22_n_langs": g("cv22", "n_langs"),
        "cv22_per_lang": {k: v["cer"] for k, v in (g("cv22", "per_lang") or {}).items()},
    }


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def versus(run: dict, base: dict, arm: str, tol: float = 0.01) -> dict:
    """How much multilingual accuracy a fine-tune keeps, language by language, against base."""
    a, b = run.get(f"{arm}_per_lang") or {}, base.get(f"{arm}_per_lang") or {}
    common = sorted(set(a) & set(b))
    if not common:
        return {}
    d = {k: a[k] - b[k] for k in common}
    import statistics
    return {"n_langs": len(common),
            "better": sum(v < -tol for v in d.values()), "same": sum(abs(v) <= tol for v in d.values()),
            "worse": sum(v > tol for v in d.values()),
            "median_delta": round(statistics.median(d.values()), 4),
            "macro_base": round(statistics.mean(b[k] for k in common), 4),
            "macro_run": round(statistics.mean(a[k] for k in common), 4),
            "worst": sorted(d.items(), key=lambda kv: -kv[1])[:5]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="base_large_v3", help="the run every other is compared with")
    args = ap.parse_args()
    out = {}
    for run in sorted(p for p in args.runs.iterdir() if (p / "eval.json").exists()):
        meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
        ww, n_ww = wild_words(run)
        cfg = {k: meta.get(k) for k in ("mix", "method", "lr", "steps", "lora_r", "lora_alpha", "lora_target_modules",
                                        "batch_size", "grad_accum", "n_clips", "blank_share", "lid_loss",
                                        "lid_weight", "blank_lang")}
        # how the benchmark decoded it: "none" = as vLLM serves (asr/scripts/bench_protocol.py)
        modes = set(bench_modes(run).values()) - {"missing"}
        cfg["decoding"] = modes.pop() if len(modes) == 1 else "mixed: " + ", ".join(sorted(modes))
        mixes = Path(meta["mix"]).parent / "summary.json" if meta.get("mix") else None
        if mixes and mixes.exists():           # asr/train/build_mix.py: clips, hours, languages, arms per mix
            ms = json.loads(mixes.read_text())
            cfg["mix_stats"] = ms.get(Path(meta["mix"]).stem)
            cfg["cv22_pool_languages"] = (ms.get("cv22_only") or {}).get("languages")
        out[run.name] = {"config": cfg,
                         **pick(json.loads((run / "eval.json").read_text())), "wild_words": ww, "wild_n": n_ww,
                         "empty_on_speech": empty_on_speech(run)}
    if args.base in out:
        for name, r in out.items():
            if name != args.base:
                r["vs_base"] = {arm: versus(r, out[args.base], arm) for arm in ("cv22", "fleurs")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    cols = ("silence_hr", "music_hr", "nonspeech_hr", "wild_words", "lexicon_recovered", "librispeech_wer",
            "genuine_wer", "fleurs_cer_macro", "cv22_cer_macro")
    print(f"{'run':<44}" + "".join(f"{c[:14]:>16}" for c in cols))
    for name, r in out.items():
        print(f"{name:<44}" + "".join(f"{r[c] if r[c] is not None else '-':>16}" for c in cols))
        for arm, v in (r.get("vs_base") or {}).items():
            if v:
                print(f"    {arm}: {v['better']} better / {v['same']} same / {v['worse']} worse of {v['n_langs']} "
                      f"languages, median delta {v['median_delta']:+.4f}, macro {v['macro_base']} -> {v['macro_run']}")


if __name__ == "__main__":
    main()
