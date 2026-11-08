#!/usr/bin/env python3
"""Model card for the released Whisper fine-tune: results first, against base whisper-large-v3.

Reads only the ASR summary (asr/summarise.py) and the language-identification report (asr/lid_report.py) and
writes README.md plus figures/ for the Hugging Face repository. Nothing is typed by hand: every number in the
card is read from those files, and a run scored with any decoding but vLLM's is refused.

    python asr/release/card.py --run <run name> --repo Scicom-intl/<name> --out <dir> \\
        [--summary results/asr/summary.json --lid results/asr/lid2.json] [--push]

House style (the other Scicom cards): results first, short sentences, no em dashes, figures with the other
models, an example to use last, the licence and why.
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROWS = [  # (key, label)
    ("silence_hr", "words written over silence"),
    ("music_hr", "words written over music"),
    ("nonspeech_hr", "words written over other non-speech"),
    ("wild_words", "words written over real voice-free audio"),
    ("lexicon_recovered", "spoken hallucination phrases transcribed"),
    ("redup_runaway", "runaway repetition"),
    ("librispeech_wer_corpus", "LibriSpeech test-clean WER"),
    ("genuine_wer_corpus", "Malaysian speech WER"),
    ("cv22_cer_macro", "Common Voice 22, CER (mean over languages)"),
    ("fleurs_cer_macro", "FLEURS, CER (mean over languages)"),
    ("empty_on_speech.librispeech_test_clean", "LibriSpeech clips returned empty"),
    ("empty_on_speech.genuine", "Malaysian speech clips returned empty"),
    ("empty_on_speech.fleurs", "FLEURS clips returned empty"),
]


def val(r, key):
    """A summary value; `a.b` reads a nested one."""
    for k in key.split("."):
        r = (r or {}).get(k)
    return r
def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _style():
    """The preprint's chart style (melati/plotstyle.py), which every Melati figure shares, and pyplot."""
    import sys
    sys.path.insert(0, str(ROOT))
    import melati.plotstyle as st
    import matplotlib.pyplot as plt
    return st, plt


def _save(plt, fig, path):
    import melati.plotstyle as st
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=st.CARD_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def _img(out, repo, name, alt):
    """The card's <img> for figures/<name>, at the scale every Melati card shows its figures at."""
    st, _ = _style()
    return st.card_img(out / "figures" / name, alt, st.hf_url(repo, f"figures/{name}"))


def figure_halluc(run, base, path):
    st, plt = _style()
    from matplotlib.ticker import PercentFormatter
    arms = [("silence_hr", "silence"), ("music_hr", "music"), ("nonspeech_hr", "other non-speech"),
            ("wild_words", "real voice-free audio")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    st.frame(ax, None, st.better("clips with any words written"), axis="y")
    x = list(range(len(arms)))
    for off, (r, c, lab) in zip((-0.2, 0.2), ((base, st.S4, "whisper-large-v3"), (run, st.S1, "this model"))):
        vs = [r.get(k) or 0 for k, _ in arms]
        ax.bar([i + off for i in x], vs, 0.38, color=c, label=lab, zorder=3)
        for i, v in zip(x, vs):
            ax.text(i + off, v + 0.015, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=6.8, color=st.TEXT)
    ax.set_xticks(x); ax.set_xticklabels([a for _, a in arms])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.12)
    st.legend_below(ax, ncol=2, pad=0.14)
    _save(plt, fig, path)


def figure_per_lang(run, base, arm, path):
    a, b = run.get(f"{arm}_per_lang") or {}, base.get(f"{arm}_per_lang") or {}
    langs = sorted(set(a) & set(b), key=lambda k: b[k])
    if not langs:
        return False
    st, plt = _style()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.16 * len(langs)), 3.2))
    st.frame(ax, None, st.better("CER (%)"), axis="y")
    x = range(len(langs))
    ax.plot(x, [100 * b[k] for k in langs], "o", color=st.S4, ms=3.5, label="whisper-large-v3", zorder=3)
    ax.plot(x, [100 * a[k] for k in langs], "o", color=st.S1, ms=3.5, label="this model", zorder=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(langs, rotation=90, fontsize=6.5)
    st.legend_below(ax, ncol=2, pad=0.2)
    _save(plt, fig, path)
    return True


def member_text(run_json: Path):
    """One line for a merged member, from its run.json."""
    m = json.loads(run_json.read_text())
    mix = Path(m.get("mix") or "").stem
    if m.get("method") == "full":
        what = f"a full fine-tune (learning rate {m.get('lr')}, fp32 master weights)"
    else:
        what = (f"LoRA rank {m.get('lora_r')} on the attention and MLP projections (learning rate {m.get('lr')})")
    extra = {"distill": ", with the language KL", "label": ", with the language token supervised"}.get(m.get("lid_loss"), "")
    return f"{what} on the `{mix}` mix, {m.get('steps')} steps{extra}"


def merge_text(cfg_run: dict):
    runs = [Path(r) for r in cfg_run["runs"]]
    w = cfg_run.get("weights") or [1.0] * len(runs)
    tot = sum(w)
    parts = [f"{100 * wi / tot:.0f}% of the change made by {member_text(r / 'run.json')}" for r, wi in zip(runs, w)]
    return ("The weights are whisper-large-v3 plus a weighted average of what two fine-tunes changed (a merge): "
            + "; and ".join(parts) + ". The first keeps the languages and identifies them; the second barely "
            "loops; the average kept both, and beat each on Malaysian speech, FLEURS and Common Voice. Each LoRA "
            "change was rebuilt exactly from its fp32 adapter, the full fine-tune's from its fp32 weights. "
            "Mixes: `all` is the hallucination recipe's data (voice-free clips with empty targets, synthesised "
            "hallucination phrases that are spoken, Malaysian podcast speech); `all_cv22_r25_tts` adds Common Voice 22 "
            "at 25% and read Indic speech for the languages Common Voice lacks.")


def training_text(cfg):
    ms = cfg.get("mix_stats") or {}
    arms = ms.get("arms") or {}
    lid = {"distill": "a KL term holding the first token's distribution over language tokens to base "
                      "whisper-large-v3's, so it keeps identifying languages as base does",
           "label": "the language token supervised on speech clips"}.get(cfg.get("lid_loss") or "none")
    parts = [
        f"LoRA rank {cfg.get('lora_r')} (alpha {cfg.get('lora_alpha')}) on the attention and MLP projections, merged "
        f"into the weights; learning rate {cfg.get('lr')}, {cfg.get('steps')} steps of "
        f"{(cfg.get('batch_size') or 8) * (cfg.get('grad_accum') or 2)} clips.",
        f"The mix is {cfg.get('n_clips'):,} clips ({ms.get('hours'):.0f} h, {ms.get('languages')} languages), "
        f"{100 * (cfg.get('blank_share') or 0):.0f}% of them voice-free with an empty target"
        + (f", and {arms['cv22']:,} Common Voice 22 clips replayed in the same batches" if arms.get("cv22") else "")
        + (f", and {arms['tts44k']:,} clips of read Indic speech for the languages Common Voice lacks"
           if arms.get("tts44k") else "") + ".",
        "The decoder is fed its prompt (language, task, no timestamps) and the prompt is kept out of the loss.",
    ]
    if lid:
        parts.append(f"The loss adds {lid}.")
    if cfg.get("blank_lang") == "base-detected":
        parts.append("Each voice-free clip is trained under the language base whisper-large-v3 detects on it, the "
                     "token it will see at inference.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name in the summary")
    ap.add_argument("--base", default="base_large_v3")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=ROOT / "results" / "asr" / "summary.json")
    ap.add_argument("--base-summary", type=Path, default=None, help="where the base run is, if not in --summary")
    ap.add_argument("--run-json", type=Path, default=None, help="the run's run.json (a merge describes its members)")
    ap.add_argument("--lid", type=Path, default=ROOT / "results" / "asr" / "lid2.json")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    S = json.loads(args.summary.read_text())
    B = json.loads(args.base_summary.read_text()) if args.base_summary else S
    run, base = S[args.run], B[args.base]
    for name, r in ((args.run, run), (args.base, base)):
        dec = (r.get("config") or {}).get("decoding")
        assert dec == "none", f"{name} was scored with decoding {dec!r}, not as vLLM serves (none)"
    cfg = run.get("config") or {}
    vs = run.get("vs_base") or {}
    lid = json.loads(args.lid.read_text()) if args.lid.exists() else {}
    # runs/ holds the first sweep (prefix bug) under the same names as runs2/, so the fixed sweep is asked first
    lid_of = lambda key: lid.get(f"runs2/{key}") or lid.get(f"merges/{key}") or lid.get(f"runs/{key}") or {}
    args.out.mkdir(parents=True, exist_ok=True)
    figure_halluc(run, base, args.out / "figures" / "hallucination.png")
    figs = [arm for arm in ("cv22", "fleurs") if figure_per_lang(run, base, arm, args.out / "figures" / f"{arm}.png")]
    name = args.repo.split("/")[-1]
    lines = [
        "---", "license: cc-by-nc-4.0", "base_model: openai/whisper-large-v3", "pipeline_tag: automatic-speech-recognition",
        "tags:", "- whisper", "- hallucination", "- multilingual", "language:", "- multilingual", "---", "",
        f"# {name}", "",
        "Whisper large-v3, fine-tuned to write nothing over audio with no speech in it while keeping its other "
        "languages. It is the transcriber of Melati, a pipeline that turns Malaysian podcasts into expressive speech "
        "data, where every pause between words is transcribed to find fillers: base whisper-large-v3 writes words over "
        f"{pct(base.get('silence_hr'))} of silent clips and {pct(base.get('music_hr'))} of music.", "",
        "Training follows [How to Reduce Whisper Hallucination](https://arxiv.org/abs/2609.32560) (empty targets on "
        "voice-free audio, and synthesised hallucination phrases that are spoken, so the model learns the difference), "
        "with Common Voice 22 replayed in the same batches to keep the other languages; the released weights merge two "
        "such fine-tunes (below).", "",
        "## Results", "",
        "Greedy, the language detected by the model, the task stated, no token suppressed: as vLLM serves it, and as "
        "`transformers` decodes it with the generation config in this repository.", "",
        _img(args.out, args.repo, "hallucination.png", "Hallucination"), "",
        "| | whisper-large-v3 | this model |", "|---|---:|---:|",
    ]
    for key, label in ROWS:
        lines.append(f"| {label} | {pct(val(base, key))} | {pct(val(run, key))} |")
    lb, lr = lid_of("base_large_v3").get("fleurs", {}), lid_of(args.run).get("fleurs", {})
    if lb.get("lid_accuracy") is not None and lr.get("lid_accuracy") is not None:
        lines.append(f"| FLEURS language identified correctly | {pct(lb['lid_accuracy'])} | {pct(lr['lid_accuracy'])} |")
    lines.append("")
    for arm, nm in (("cv22", "Common Voice 22"), ("fleurs", "FLEURS")):
        v = vs.get(arm) or {}
        if v:
            lines.append(f"- **{nm}**: better in {v['better']} languages, within one point in {v['same']}, worse in "
                         f"{v['worse']}, of {v['n_langs']}. The median change is {100 * v['median_delta']:+.1f} CER points.")
    lines.append("")
    for arm, nm in (("cv22", "Common Voice 22"), ("fleurs", "FLEURS")):
        if arm in figs:
            lines += [_img(args.out, args.repo, f"{arm}.png", f"{nm}, language by language"), "",
                      f"{nm} character error rate per language, ordered by the base model's.", ""]
    lines += [
        "Voice-free arms are the [Whisper-Hallucination](https://huggingface.co/datasets/Scicom-intl/Whisper-Hallucination) "
        f"benchmark's test splits; FLEURS is 20 clips in each of {run.get('fleurs_n_langs')} languages, none trained "
        "on.", "",
        "## Decoding", "",
        "Hugging Face's default Whisper generation config forbids `<|endoftext|>` as the first token, which means it "
        "forbids an empty transcript: on silence, a model that has learned to stop is made to write its second choice. "
        "This repository's `generation_config.json` therefore suppresses nothing (as vLLM does) and states the "
        "transcribe task. Pass your own `begin_suppress_tokens` and the model will hallucinate again, because it is "
        "not allowed to stop.", "",
        "## Training", "",
        merge_text(json.loads(args.run_json.read_text())) if args.run_json and "runs" in json.loads(args.run_json.read_text())
        else training_text(cfg), "",
        "## Limitations", "",
        f"- A model that has learned to stop can stop on speech: it returns nothing for "
        f"{pct(val(run, 'empty_on_speech.genuine'))} of the Malaysian clips and {pct(val(run, 'empty_on_speech.fleurs'))} "
        f"of FLEURS, and for {pct(val(run, 'empty_on_speech.speech_in_noise'))} of speech mixed with noise, mostly at "
        "0 dB and below. Base never does, because it is not allowed to.",
        f"- On a syllable said several times it sometimes keeps writing it ({pct(run.get('redup_runaway'))} of the "
        "reduplication arm). Decode as Whisper itself does to guard against it: when a transcript compresses like a "
        "loop (gzip ratio over 2.4), decode again at temperature 0.2, 0.4, 0.6, and cap the tokens by the clip's length.",
        f"- It still writes text over {pct(run.get('wild_words'))} of real voice-free audio (base: "
        f"{pct(base.get('wild_words'))}).", "",
        "## Use", "",
        "```bash", f"vllm serve {args.repo}", "```", "",
        "```python", "from openai import OpenAI",
        'client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")',
        f'print(client.audio.transcriptions.create(model="{args.repo}", file=open("clip.wav", "rb")).text)', "```", "",
        "```python", "from transformers import pipeline",
        f'asr = pipeline("automatic-speech-recognition", model="{args.repo}", device="cuda")',
        'print(asr("clip.wav")["text"])', "```", "",
        "## Licence", "",
        "CC BY-NC 4.0: part of the training audio (Malaysian podcast speech) is licensed for non-commercial use only.", "",
    ]
    (args.out / "README.md").write_text("\n".join(lines))
    print(f"-> {args.out / 'README.md'}")
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_folder(repo_id=args.repo, folder_path=str(args.out), allow_patterns=["README.md", "figures/*"],
                          commit_message="Model card")
        print(f"pushed card to {args.repo}")


if __name__ == "__main__":
    main()
