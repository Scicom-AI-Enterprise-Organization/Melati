#!/usr/bin/env python3
"""Decode benchmark arms with vLLM itself, the way its transcription endpoint does. RUNS ON THE BOX (.venv-diar).

run_benchmark.py imitates vLLM (no suppressed tokens, the task stated). This checks the imitation on the real
server code: per clip, the language is one token from `<|startoftranscript|>` constrained to the language
tokens (vLLM's get_language_detection_prompt + allowed_token_ids), then the transcript from
`<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>`, greedy (get_generation_prompt). Arms come from
dump_arms.py. Rows are written in run_benchmark.py's layout, and with --hf it reports how often the two
decoders wrote the same transcript.

    python asr/bench/vllm_check.py --model <merged dir> --arms-dir $MELATI_DATA/asr/arms_npz \\
        --arms silence music nonspeech fleurs --out <dir> [--hf <run_benchmark output dir>]
"""
import argparse, json, time
from pathlib import Path

import numpy as np

from metrics import cer                 # noqa: E402
from metrics import normalise as norm   # the benchmark's own normaliser (dependency-free)

SR = 16000


def load_arm(d: Path, arm: str):
    z = np.load(d / f"{arm}.npz")
    a, off = z["audio"], z["offsets"]
    meta = [json.loads(l) for l in (d / f"{arm}.jsonl").open(encoding="utf-8")]
    return [a[off[i]:off[i + 1]] for i in range(len(meta))], meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms-dir", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf", type=Path, default=None, help="run_benchmark.py's output dir for the same model")
    ap.add_argument("--max-new-tokens", type=int, default=440)
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--dtype", default="float16",
                    help="float16 matches run_benchmark.py; the pipeline serves in bfloat16 (serve/whisper_vllm.sh)")
    args = ap.parse_args()

    from transformers import GenerationConfig
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=448, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"audio": 1}, enforce_eager=False)
    tok = llm.get_tokenizer()
    gc = GenerationConfig.from_pretrained(args.model)
    lang_ids = sorted(gc.lang_to_id.values())                       # the same set HF's detect_language uses
    detect = SamplingParams(temperature=0.0, max_tokens=1, allowed_token_ids=lang_ids)
    write = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, skip_special_tokens=True)
    tag = Path(args.model.rstrip("/")).name + ("" if args.dtype == "float16" else f"_{args.dtype}")
    summary = {}
    for arm in args.arms:
        waves, meta = load_arm(args.arms_dir, arm)
        t0 = time.time()
        enc = [{"prompt": "", "multi_modal_data": {"audio": (w, SR)}} for w in waves]
        det = llm.generate([{"encoder_prompt": e, "decoder_prompt": "<|startoftranscript|>"} for e in enc], detect,
                           use_tqdm=False)
        langs = [tok.convert_ids_to_tokens(o.outputs[0].token_ids[0]).strip("<|>") for o in det]
        gen = llm.generate([{"encoder_prompt": e,
                             "decoder_prompt": f"<|startoftranscript|><|{l}|><|transcribe|><|notimestamps|>"}
                            for e, l in zip(enc, langs)], write, use_tqdm=False)
        hyps = [o.outputs[0].text.strip() for o in gen]
        el = time.time() - t0
        out = args.out / tag / f"{arm}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for m, h, l in zip(meta, hyps, langs):
                fh.write(json.dumps({"audio_filepath": m["id"], "arm": arm, "hyp": h, "lid": l,
                                     "forced_language": False, "suppress": "vllm",
                                     "reference_text": m["reference_text"], "meta": {"lang": m["lang"]}},
                                    ensure_ascii=False) + "\n")
        s = {"n": len(hyps), "seconds": round(el, 1), "clips_per_s": round(len(hyps) / max(el, 1e-9), 1),
             "words_rate": round(sum(bool(norm(h)) for h in hyps) / len(hyps), 4)}
        if args.hf:
            fs = sorted(args.hf.glob(f"*/{arm}.jsonl"))
            if fs:
                ref = {r["audio_filepath"]: r for r in map(json.loads, fs[0].open(encoding="utf-8"))}
                pairs = [(ref[m["id"]], h, l) for m, h, l in zip(meta, hyps, langs) if m["id"] in ref]
                s["hf_n"] = len(pairs)
                s["same_transcript"] = round(sum(norm(r["hyp"]) == norm(h) for r, h, _ in pairs) / len(pairs), 4)
                s["same_language"] = round(sum((r.get("lid") or l) == l for r, _, l in pairs) / len(pairs), 4)
                s["hf_words_rate"] = round(sum(bool(norm(r["hyp"])) for r, _, _ in pairs) / len(pairs), 4)
                if any(r.get("reference_text") for r, _, _ in pairs):   # speech: CER of both, mean over languages
                    by = {}
                    for (r, h, _), m in zip(pairs, [m for m in meta if m["id"] in ref]):
                        if r.get("reference_text"):
                            by.setdefault(m["lang"], []).append((cer(r["reference_text"], h),
                                                                 cer(r["reference_text"], r["hyp"])))
                    s["cer_macro_vllm"] = round(float(np.mean([np.mean([a for a, _ in v]) for v in by.values()])), 4)
                    s["cer_macro_hf"] = round(float(np.mean([np.mean([b for _, b in v]) for v in by.values()])), 4)
                    s["n_langs"] = len(by)
        summary[arm] = s
        print(f"[vllm] {tag}/{arm}: {json.dumps(s)}", flush=True)
    (args.out / tag / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
