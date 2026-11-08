#!/usr/bin/env python3
"""Run a Whisper checkpoint over the Whisper-Hallucination benchmark.

Pulls the benchmark straight from the Hub, so this is exactly the path anyone reproducing
the numbers would take -- no local prep, no private data.

    python run_benchmark.py --model openai/whisper-large-v3 --device cuda:6

Decoding is deliberately plain: greedy, no forced language, no temperature fallback, and no
suppressed tokens, which is how vLLM serves Whisper (vllm_check.py runs vLLM itself to confirm
it). The model's default suppression lists are dropped unless --hf-suppress is given:
begin_suppress_tokens forbids <|endoftext|> as the first token, so under Hugging Face's
default a model that has learned to stop on silence is made to write something anyway. The
point is to measure what the CHECKPOINT does as served, not what a decoding wrapper adds or
papers over. Any mitigation is a separate arm.

Language is auto-detected rather than forced, including on the speech arms, because
language drift on low-evidence audio is part of the failure being measured.

Audio is read with soundfile via `Audio(decode=False)`; `datasets>=5` otherwise demands
torchcodec just to hand back an array.
"""
import argparse, io, json, os, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

REPO = "Scicom-intl/Whisper-Hallucination"
FLEURS_REPO = "google/fleurs"
FLEURS_PER_LANG = 20          # 20 per language; ~44 of the 60 resolve

FLEURS = {
    "lt": "lt_lt", "az": "az_az", "ne": "ne_np", "am": "am_et", "my": "my_mm",
    "fi": "fi_fi", "sk": "sk_sk", "kn": "kn_in", "th": "th_th", "lo": "lo_la",
    "cs": "cs_cz", "ja": "ja_jp", "pl": "pl_pl", "ta": "ta_in", "ur": "ur_pk",
    "el": "el_gr", "vi": "vi_vn", "ms": "ms_my", "id": "id_id", "en": "en_us",
    "fr": "fr_fr", "de": "de_de", "es": "es_419", "pt": "pt_br", "ar": "ar_eg",
    "zh": "cmn_hans_cn", "ko": "ko_kr", "hi": "hi_in", "km": "km_kh", "sw": "sw_ke",
    "si": "si_lk", "sd": "sd_in", "te": "te_in", "ml": "ml_in", "mg": "mg_mg",
    "yo": "yo_ng", "pa": "pa_in", "gu": "gu_in", "mr": "mr_in", "bn": "bn_in",
    "ka": "ka_ge", "hy": "hy_am", "kk": "kk_kz", "mn": "mn_mn", "ps": "ps_af",
    "sn": "sn_zw", "so": "so_so", "ha": "ha_ng", "yue": "yue_hant_hk", "uz": "uz_uz",
    "sl": "sl_si", "et": "et_ee", "ro": "ro_ro", "da": "da_dk", "bg": "bg_bg",
    "sv": "sv_se", "hu": "hu_hu", "nl": "nl_nl", "it": "it_it", "tr": "tr_tr",
}
ARMS = ["silence", "music", "nonspeech", "reduplication", "speech_in_noise",
        "genuine", "genuine_isolated", "librispeech_test_clean",
        # Not a benchmark arm -- synthetic positives. Its `test` split is the contrastive
        # other half of `silence`/`music`/`nonspeech`: the SAME phrases, but actually spoken.
        # Emitting the phrase here is the correct answer, so it measures what a hallucination
        # filter would wrongly delete.
        "lexicon_synth",
        # Real audio that made a model fail. Its `test` split only -- `wild` train exists and
        # is used by the fine-tuning sweep, so benchmarking the train half would be scoring a
        # model on what it learned from.
        "wild",
        # Multilingual accuracy guard. librispeech only watches English, while the training
        # corpus spans 75+ languages -- a mix that quietly wrecked Tamil would pass unnoticed.
        "fleurs",
        # Melati: Common Voice 22's own test split, 20 clips per language. IN-domain for the
        # replay mixes (they train on CV22 train), so FLEURS stays the honest guard and this
        # one says whether replay held the languages it was given.
        "cv22"]
SR = 16000




def load_fleurs(per_lang: int, limit: int = 0, cache: Path = None):
    """FLEURS test, streamed per language and sampled, then CACHED to one parquet.

    `datasets` cannot give a multilingual slice in one call, so each config is streamed and
    truncated. Streaming also avoids pulling the whole 102-language corpus for ~900 clips --
    but it would re-pull them for every checkpoint in a 12-run sweep, and two runs that
    streamed different clips would not be comparable. So the first call materialises the
    sample and every later one reads it back.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if cache and Path(cache).exists():
        tb = pq.read_table(cache).to_pylist()
        rows = [{"id": r["id"], "lang": r["lang"],
                 "audio": {"bytes": r["audio_bytes"], "path": r["id"]},
                 "reference_text": r["reference_text"]} for r in tb]
        print(f"[fleurs] {len(rows)} cached clips over "
              f"{len({r['lang'] for r in rows})} languages -> {cache}", flush=True)
        return rows[:limit] if limit else rows

    from datasets import Audio, load_dataset

    rows = []
    for lang, code in FLEURS.items():
        try:
            try:
                ds = load_dataset(FLEURS_REPO, code, split="test", streaming=True)
                ds = ds.cast_column("audio", Audio(decode=False))
                ds = iter(ds)
            except Exception:
                ds = fleurs_parquet_rows(code)
            for i, r in enumerate(ds):
                if i >= per_lang:
                    break
                a = r["audio"]
                raw = a.get("bytes") if isinstance(a, dict) else None
                if raw is None:
                    continue
                rows.append({"id": f"fleurs_{lang}_{i:03d}", "lang": lang,
                             "audio": {"bytes": raw, "path": f"{lang}_{i}"},
                             "reference_text": r.get("transcription") or r.get("raw_transcription") or ""})
            print(f"  [fleurs] {lang} ({code}): {len(rows)} clips so far", flush=True)
        except Exception as e:
            print(f"  [fleurs] {lang} ({code}) skipped: {type(e).__name__} {str(e)[:60]}", flush=True)
        if limit and len(rows) >= limit:
            break
    print(f"[fleurs] {len(rows)} clips over {len({r['lang'] for r in rows})} languages", flush=True)
    if cache:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a sweep runs several checkpoints at once and two of them can
        # reach this line together. A half-written parquet read by the other is worse than
        # a redundant download.
        tmp = cache.with_suffix(f".parquet.{os.getpid()}.part")
        pq.write_table(pa.table({
            "id": [r["id"] for r in rows], "lang": [r["lang"] for r in rows],
            "reference_text": [r["reference_text"] for r in rows],
            "audio_bytes": [r["audio"]["bytes"] for r in rows],
        }), tmp, compression="zstd", use_dictionary=[])   # audio is unique + compressed
        tmp.rename(cache)
        print(f"[fleurs] cached -> {cache}", flush=True)
    return rows[:limit] if limit else rows


def fleurs_parquet_rows(code: str):
    """FLEURS rows straight from `parquet-data/<code>/test-*.parquet`, one row group at a time.

    Melati: with `datasets` 4 some FLEURS configs no longer resolve through `load_dataset`
    (the paper's run got 44 of 60). The repo ships the same test split as parquet, and pyarrow
    over HfFileSystem reads only the row groups it is asked for, not the 500 MB file."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    for path in sorted(fs.glob(f"datasets/{FLEURS_REPO}/parquet-data/{code}/test-*.parquet")):
        pf = pq.ParquetFile(fs.open(path, block_size=8 << 20))
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g).to_pylist():
                yield r


def load_cv22_test(manifest: Path, limit: int = 0):
    """The CV22 test sample written by asr/scripts/build_cv22.py (16 kHz FLAC + manifest)."""
    import csv
    rows = []
    for i, r in enumerate(csv.DictReader(Path(manifest).open(encoding="utf-8"))):
        raw = (Path(manifest).parent / r["audio_filepath"]).read_bytes()
        rows.append({"id": f"cv22_{r['lang']}_{Path(r['audio_filepath']).stem}", "lang": r["lang"],
                     "audio": {"bytes": raw, "path": r["audio_filepath"]},
                     "reference_text": r["text"]})
    print(f"[cv22] {len(rows)} clips over {len({r['lang'] for r in rows})} languages", flush=True)
    return rows[:limit] if limit else rows


def reference_of(row: dict, arm: str) -> str:
    if arm in ("silence", "music", "nonspeech"):
        return ""                                   # correct output is nothing at all
    if arm == "lexicon_synth":
        return row.get("phrase") or ""              # the phrase the clip actually says
    if arm == "wild":
        # Mixed by design: a HALAS clip carries a human-corrected transcript, a blank_speech
        # clip carries nothing because the correct output IS nothing. `score_eval_run.py`
        # splits them on `reasons` rather than averaging the two together.
        return row.get("reference_text") or ""
    return row.get("reference_text") or row.get("text") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=440)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pad-lead", type=float, default=0.0,
                    help="seconds of silence prepended to every clip")
    ap.add_argument("--pad-tail", type=float, default=0.0,
                    help="seconds of silence appended to every clip")
    ap.add_argument("--pad-kind", choices=["zeros", "roomtone"], default="zeros",
                    help="`zeros` is digital silence -- note Whisper already zero-pads every "
                         "clip to a 30 s window, so TRAILING zeros change nothing by "
                         "construction and leading zeros only shift where the speech starts. "
                         "`roomtone` pads with real near-silence from the `silence` arm, which "
                         "is a different signal entirely: a noise floor Whisper can hallucinate "
                         "on. Use zeros as the control and roomtone as the realistic case.")
    ap.add_argument("--fleurs-per-lang", type=int, default=FLEURS_PER_LANG)
    ap.add_argument("--fleurs-cache", type=Path, default=Path("bench/fleurs_sample.parquet"),
                    help="built on first use; every later run reads the same clips")
    ap.add_argument("--cv22-test", type=Path,
                    default=Path(f"{MELATI_DATA}/asr/cv22/test/manifest.csv"),
                    help="CV22 test sample (asr/scripts/build_cv22.py)")
    ap.add_argument("--hf-suppress", action="store_true",
                    help="keep Whisper's suppress_tokens and begin_suppress_tokens. Off by default: vLLM, which serves "
                         "the model, applies neither, and begin_suppress_tokens forbids <|endoftext|> as the first token, "
                         "so with the prompt stated a model cannot return an empty transcript at all")
    ap.add_argument("--force-language", action="store_true",
                    help="decode each clip with its own language token (meta `lang`, fleurs/cv22): "
                         "transcription measured apart from language identification")
    ap.add_argument("--pad-tag", default="",
                    help="suffix for the results filename, so padded runs do not overwrite the "
                         "unpadded ones (e.g. lexicon_synth__tone2.jsonl)")
    args = ap.parse_args()

    pad_lead = np.zeros(int(args.pad_lead * SR), dtype="float32")
    pad_tail = np.zeros(int(args.pad_tail * SR), dtype="float32")
    tone = None
    if args.pad_kind == "roomtone" and (pad_lead.size or pad_tail.size):
        # Real room tone, not synthesised noise: the `silence` arm is 42 clips over 6 noise
        # floors, and it is the same material the negative arms measure hallucination on.
        sil = load_dataset(REPO, "silence", split="test").cast_column("audio", Audio(decode=False))
        chunks = []
        for r in sil:
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if sr == SR:
                chunks.append(x)
        tone = np.concatenate(chunks) if chunks else None
        print(f"[pad ] room tone pool: {tone.size/SR:.1f}s from {len(chunks)} clips", flush=True)

    rng = np.random.default_rng(0)

    def pad(x):
        if not (pad_lead.size or pad_tail.size):
            return x
        if tone is None:
            return np.concatenate([pad_lead, x, pad_tail])
        def draw(n):
            if n == 0:
                return np.zeros(0, dtype="float32")
            start = int(rng.integers(0, max(1, tone.size - n)))
            seg = tone[start:start + n]
            return np.pad(seg, (0, max(0, n - seg.size)))
        return np.concatenate([draw(pad_lead.size), x, draw(pad_tail.size)]).astype("float32")

    tag = args.model.split("/")[-1]
    torch.set_grad_enabled(False)
    print(f"[load] {args.model} -> {args.device}", flush=True)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()
    if not args.hf_suppress:                       # decode as vLLM does: no token is forbidden
        model.generation_config.suppress_tokens = None
        model.generation_config.begin_suppress_tokens = None

    for arm in args.arms:
        out = args.out / tag / f"{arm}{args.pad_tag}.jsonl"
        if out.exists() and not args.overwrite:
            print(f"[skip] {tag}/{arm}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        if arm == "fleurs":
            ds = load_fleurs(args.fleurs_per_lang, args.limit, args.fleurs_cache)
        elif arm == "cv22":
            ds = load_cv22_test(args.cv22_test, args.limit)
        else:
            ds = load_dataset(REPO, arm, split="test").cast_column("audio", Audio(decode=False))
            if args.limit:
                ds = ds.select(range(min(args.limit, len(ds))))
        padding = (f"  +pad {args.pad_lead}s/{args.pad_tail}s {args.pad_kind}"
                   if (args.pad_lead or args.pad_tail) else "")
        print(f"[run ] {tag}/{arm}{args.pad_tag}: {len(ds)} clips{padding}", flush=True)

        t0, tmp = time.time(), out.with_suffix(".jsonl.part")
        with tmp.open("w", encoding="utf-8") as fh:
            for start in range(0, len(ds), args.batch_size):
                rows = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
                waves = []
                for r in rows:
                    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    assert sr == SR, f"{r['id']} is {sr} Hz"
                    waves.append(pad(x))
                feats = processor(waves, sampling_rate=SR, return_tensors="pt",
                                  return_attention_mask=True)
                inp = feats.input_features.to(args.device, torch.float16)
                am = feats.get("attention_mask")
                kw = {}
                langs = [r.get("lang") or (r.get("meta") or {}).get("lang") for r in rows]
                if args.force_language and all(langs):
                    kw["language"] = langs                 # one per clip
                gen = model.generate(
                    inp,
                    attention_mask=am.to(args.device) if am is not None else None,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, num_beams=1,          # greedy: measure the checkpoint
                    # the task is always transcription, stated rather than left to the checkpoint's saved
                    # generation config: fine-tunes saved without forced_decoder_ids otherwise PREDICT the task
                    # token, and one that obeys its tokens then translates (Yoruba written as English); vLLM,
                    # where the model is served, always sends <|transcribe|>
                    task="transcribe",
                    **kw,
                )
                texts = processor.batch_decode(gen, skip_special_tokens=True)
                # the language the model decoded with: generate() returns the text without its prompt, so it
                # is asked directly (detect_language is what generate itself calls); forced, it is the given one
                if kw:
                    lids = list(langs)
                elif any(langs):
                    ids = model.detect_language(input_features=inp)
                    lids = [t.strip("<|>") for t in processor.tokenizer.convert_ids_to_tokens(ids.tolist())]
                else:
                    lids = [None] * len(rows)
                for r, hyp, lid in zip(rows, texts, lids):
                    fh.write(json.dumps({
                        "audio_filepath": r.get("id", ""),
                        "arm": arm,
                        "pad": {"lead_s": args.pad_lead, "tail_s": args.pad_tail,
                                "kind": args.pad_kind},
                        "hyp": hyp.strip(),
                        "lid": lid,
                        "forced_language": bool(kw),
                        "suppress": "hf" if args.hf_suppress else "none",
                        "reference_text": reference_of(r, arm),
                        "meta": {k: v for k, v in r.items() if k != "audio"},
                    }, ensure_ascii=False) + "\n")
                done = min(start + args.batch_size, len(ds))
                if done % (args.batch_size * 20) == 0 or done == len(ds):
                    el = time.time() - t0
                    print(f"   {done}/{len(ds)}  {el:.0f}s  ({done/max(el,1e-9):.1f} clip/s)", flush=True)
        tmp.rename(out)
        print(f"[done] {tag}/{arm} -> {out}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
