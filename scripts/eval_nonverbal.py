#!/usr/bin/env python3
"""Score non-verbal candidates against human-placed tags (NonverbalTTS test split), through the servers.

NonverbalTTS (deepvk/NonverbalTTS, Apache-2.0) marks ten kinds of vocalisation inline in the
transcript (an emoji at the word boundary where it happens), placed by detectors and fixed by three
annotators each; its test split is 359 clips of VoxCeleb and Expresso speech. For every clip this
asks the front server for PANNs frame tracks (/sed) and the align server for the reference
transcript's word times (/align), extracts candidate events exactly as the pipeline does
(melati.nonverbal.events_from_tracks), places each at its nearest word boundary, and scores per
class, at a sweep of probability floors:

  * clip level: does the clip contain the vocalisation at all (precision, recall)
  * placement: of the tags placed in a clip that has the class, how many land within one word
    boundary of a reference tag of that class

The lowest floor reaching --target-precision per class is what export_config.json's `sed_min_p`
takes. Breath is the commonest reference tag and the one annotators agree on least.

    python scripts/eval_nonverbal.py --endpoints $MELATI_DATA/serve_dev/endpoints.json \\
        --out results/nonverbal/nonverbaltts.json
"""
import argparse, asyncio, io, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melati import audio as A  # noqa: E402
from melati import nonverbal as NV  # noqa: E402
from melati.tags import Event, Word, event_slot  # noqa: E402

# NonverbalTTS emoji -> melati.sed group (the NV classes melati.nonverbal.TO_TAG maps to tags)
EMOJI = {"🌬️": "breath", "🌬": "breath", "😂": "laugh", "🤣": "laugh", "😮‍💨": "sigh", "🤧": "sneeze", "😷": "cough",
         "🗣️": "throat_clear", "🗣": "throat_clear", "😖": "groan", "😤": "groan", "👃": "sniff"}
EMO_RE = re.compile("|".join(sorted(map(re.escape, EMOJI), key=len, reverse=True)))


def reference(text):
    words, tags = [], []
    for tok in (text or "").split():
        for f in EMO_RE.findall(tok):
            tags.append((EMOJI[f], len(words)))
        rest = EMO_RE.sub("", tok).strip()
        if rest:
            words.append(rest)
    return words, tags


async def run(args):
    from datasets import Audio, load_dataset
    from melati.client import Client, Endpoints
    d = load_dataset("deepvk/NonverbalTTS", split="test").cast_column("audio", Audio(decode=False))
    seen = Counter(e for r in d for e in EMO_RE.findall(r["Result"] or ""))
    other = Counter(ch for r in d for ch in EMO_RE.sub("", r["Result"] or "") if ord(ch) > 0x2000 and ch != "\ufe0f")
    print("mapped:", dict(seen), "unmapped symbols:", dict(other.most_common(8)), flush=True)
    clips = []
    for r in d:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        words, tags = reference(r["Result"])
        clips.append({"x": x.mean(1), "sr": sr, "words": words, "tags": tags})
    async with Client(Endpoints(args.endpoints)) as c:
        async def one(cl):
            x32 = A.resample(cl["x"], cl["sr"], 32000)
            x16 = A.resample(cl["x"], cl["sr"], 16000)
            tr, al = await asyncio.gather(c.sed(x32), c.align([(x16, " ".join(cl["words"]), "en")]))
            return tr, al[0]
        res = await asyncio.gather(*[one(cl) for cl in clips])
    floors = [round(v, 2) for v in np.arange(0.05, 0.95, 0.05)]
    stats = {f: defaultdict(Counter) for f in floors}
    for cl, (tr, al) in zip(clips, res):
        fps = tr.pop("fps")
        dur = len(cl["x"]) / cl["sr"]
        evs = NV.events_from_tracks(tr, fps, 0.0, dur, default=0.05)
        words = [Word(w["w"], w["s"], w["e"]) for w in (al.get("words") or [])]
        ref = defaultdict(list)
        for cls, slot in cl["tags"]:
            ref[cls].append(slot)
        for f in floors:
            hyp = defaultdict(list)
            for e in evs:
                if e["label"] in NV.TO_TAG and e["p"] >= f:
                    hyp[e["label"]].append(event_slot(Event(e["label"], e["s"], e["e"]), words) if words else 0)
            for cls in set(ref) | set(hyp):
                st = stats[f][cls]
                st["clip_tp"] += bool(ref[cls]) and bool(hyp[cls])
                st["clip_fp"] += (not ref[cls]) and bool(hyp[cls])
                st["clip_fn"] += bool(ref[cls]) and not hyp[cls]
                if ref[cls]:
                    st["placed"] += len(hyp[cls])
                    st["placed_ok"] += sum(any(abs(h - r_) <= 1 for r_ in ref[cls]) for h in hyp[cls])
    out = {"n_clips": len(clips), "reference_tags": dict(Counter(t for cl in clips for t, _ in cl["tags"])),
           "detector": "PANNs Cnn14_DecisionLevelMax", "by_floor": {}, "chosen_floor": {}}
    for f in floors:
        out["by_floor"][str(f)] = {
            cls: {"precision": round(st["clip_tp"] / max(st["clip_tp"] + st["clip_fp"], 1), 4),
                  "recall": round(st["clip_tp"] / max(st["clip_tp"] + st["clip_fn"], 1), 4),
                  "placement": round(st["placed_ok"] / max(st["placed"], 1), 4),
                  "tp": st["clip_tp"], "fp": st["clip_fp"], "fn": st["clip_fn"]}
            for cls, st in stats[f].items()}
    for cls in out["reference_tags"]:
        for f in floors:
            r = out["by_floor"][str(f)].get(cls)
            if r and r["tp"] + r["fp"] >= 5 and r["precision"] >= args.target_precision:
                out["chosen_floor"][cls] = {"floor": f, **r}
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["chosen_floor"], indent=1))
    best = {cls: max(((f, out["by_floor"][str(f)].get(cls, {})) for f in floors),
                     key=lambda fr: (2 * fr[1].get("precision", 0) * fr[1].get("recall", 0)
                                     / max(fr[1].get("precision", 0) + fr[1].get("recall", 0), 1e-9)))
            for cls in out["reference_tags"]}
    print("best F1 floor per class:", {c: (f, r.get("precision"), r.get("recall")) for c, (f, r) in best.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-precision", type=float, default=0.7)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
