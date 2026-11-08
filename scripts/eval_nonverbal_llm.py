#!/usr/bin/env python3
"""Can an audio LLM place non-verbal tags? Qwen3-Omni (vLLM) against NonverbalTTS's human-placed tags.

Each NonverbalTTS test clip is sent to the `omni` service with one instruction: transcribe verbatim
and mark each non-verbal sound inline with a tag from a closed set. The tags are mapped to the
same classes eval_nonverbal.py scores, and the hypothesis words are aligned to the reference words
(difflib on normalised tokens) so a tag's position can be compared with the reference's, within one
word boundary. Scored per class at clip level (does the clip contain it) and for placement.

    python scripts/eval_nonverbal_llm.py --endpoints $MELATI_DATA/serve_eval/endpoints.json \\
        --out results/nonverbal/nonverbaltts_omni.json
"""
import argparse, asyncio, difflib, io, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from scripts.eval_nonverbal import reference  # noqa: E402

PROMPTS = {
    "plain": ("Transcribe this audio verbatim, word for word, in the language spoken. Wherever the speaker makes a "
              "non-verbal sound, insert a tag at that point, using only these tags: [laughs] [breath] [coughs] [sighs] "
              "[sniffs] [clears throat] [gasps] [cries] [groans] [sneezes] [yawns]. Do not describe anything else. "
              "Output only the tagged transcript."),
    # the first prompt wrote fillers faithfully and almost never a breath: name the breath, and say how
    # to tell one from silence, and ask for nothing that cannot be heard
    "breath": ("Transcribe this audio verbatim, word for word, in the language spoken, keeping fillers such as "
               "\"um\" and \"uh\". Mark every non-verbal sound the speaker makes at the point where it happens, "
               "with one of these tags: [breath] for an audible breath, usually an inhalation before or between "
               "phrases; [laughs] for laughter or a chuckle, including laughing while speaking; [coughs]; [sighs]; "
               "[sniffs]; [clears throat]; [gasps]; [cries]; [groans]; [sneezes]; [yawns]. Tag only sounds you can "
               "clearly hear, never guess. Output only the tagged transcript."),
    "strict": ("Listen carefully for non-verbal sounds made by the speaker: breaths, laughter, coughs, sighs, sniffs, "
               "throat clearing, gasps, crying, groans, sneezes, yawns. Transcribe the audio verbatim, word for word, "
               "and insert a tag exactly where each such sound occurs: [breath] [laughs] [coughs] [sighs] [sniffs] "
               "[clears throat] [gasps] [cries] [groans] [sneezes] [yawns]. Breaths between phrases count. Do not tag "
               "a sound that is not there. Output only the tagged transcript."),
}
TAGS = {"laugh": "laugh", "chuckl": "laugh", "giggl": "laugh", "breath": "breath", "inhal": "breath",
        "exhal": "breath", "cough": "cough", "sigh": "sigh", "sniff": "sniff", "throat": "throat_clear",
        "gasp": "gasp", "cry": "cry", "cries": "cry", "sob": "cry", "groan": "groan", "grunt": "groan",
        "sneez": "sneeze", "yawn": "yawn"}
TAG_RE = re.compile(r"\[([^\]]+)\]")
TOK = re.compile(r"[\w']+")


def parse(text):
    """-> (words, [(class, boundary index)])"""
    words, tags = [], []
    for m in re.finditer(r"\[[^\]]+\]|[^\s\[\]]+", text):
        tok = m.group(0)
        if tok.startswith("["):
            t = tok[1:-1].lower()
            cls = next((v for k, v in TAGS.items() if k in t), None)
            if cls:
                tags.append((cls, len(words)))
        else:
            w = "".join(TOK.findall(tok.lower()))
            if w:
                words.append(w)
    return words, tags


def map_slots(hyp_words, ref_words):
    """hypothesis boundary index -> reference boundary index, through a word alignment."""
    sm = difflib.SequenceMatcher(a=hyp_words, b=ref_words, autojunk=False)
    m = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size + 1):
            m.setdefault(a + k, b + k)
    keys = sorted(m)
    return lambda i: m[min(keys, key=lambda k: abs(k - i))] if keys else 0


async def run(args):
    from datasets import Audio, load_dataset
    from melati.client import Client, Endpoints
    d = load_dataset("deepvk/NonverbalTTS", split="test").cast_column("audio", Audio(decode=False))
    clips = []
    for r in d:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        words, tags = reference(r["Result"])
        norm = ["".join(TOK.findall(w.lower())) for w in words]
        # a reference tag's slot counts raw tokens; re-count it over the words that survive
        # normalisation, which are the ones the hypothesis is aligned to
        kept = [sum(1 for w in norm[:k] if w) for k in range(len(norm) + 1)]
        clips.append({"x": A.resample(x.mean(1), sr, 16000), "words": [w for w in norm if w],
                      "tags": [(cls, kept[slot]) for cls, slot in tags]})
    async with Client(Endpoints(args.endpoints)) as c:
        outs = await asyncio.gather(*[c.chat_audio("omni", cl["x"], PROMPTS[args.prompt], 400) for cl in clips])
    stats = defaultdict(Counter)
    examples = []
    for cl, out in zip(clips, outs):
        hw, ht = parse(out)
        to_ref = map_slots(hw, cl["words"])
        ref, hyp = defaultdict(list), defaultdict(list)
        for cls, slot in cl["tags"]:
            ref[cls].append(slot)
        for cls, slot in ht:
            hyp[cls].append(to_ref(slot))
        for cls in set(ref) | set(hyp):
            st = stats[cls]
            st["tp"] += bool(ref[cls]) and bool(hyp[cls])
            st["fp"] += (not ref[cls]) and bool(hyp[cls])
            st["fn"] += bool(ref[cls]) and not hyp[cls]
            if ref[cls]:
                st["placed"] += len(hyp[cls])
                st["placed_ok"] += sum(any(abs(h - r_) <= 1 for r_ in ref[cls]) for h in hyp[cls])
        if len(examples) < 12 and (cl["tags"] or ht):
            examples.append({"hyp": out[:300], "ref_tags": cl["tags"]})
    res = {cls: {"precision": round(st["tp"] / max(st["tp"] + st["fp"], 1), 4),
                 "recall": round(st["tp"] / max(st["tp"] + st["fn"], 1), 4),
                 "placement": round(st["placed_ok"] / max(st["placed"], 1), 4), **dict(st)} for cls, st in stats.items()}
    out = {"n_clips": len(clips), "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "prompt_name": args.prompt,
           "prompt": PROMPTS[args.prompt], "per_class": res}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if args.outputs:    # each clip's answer beside NonverbalTTS's own words (CC BY-NC-SA 4.0), so never into results/
        args.outputs.parent.mkdir(parents=True, exist_ok=True)
        args.outputs.write_text(json.dumps(
            {"examples": examples,
             "outputs": [{"hyp": o, "ref_tags": cl["tags"], "ref_words": cl["words"]} for cl, o in zip(clips, outs)]},
            indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, help="also write what the model said for every clip here (outside results/)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="plain")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
