#!/usr/bin/env python3
"""AudioSet's frame-level taggers over Melati's segments: the average of PretrainedSED's BEATs, ATST-F and M2D
(fine-tuned on AudioSet's strong labels; nonverbal/audioset_eval.py measured them class by class), every 0.1 s of
every segment, for each vocal sound and delivery Melati has a tag for (melati.nonverbal.TO_TAG, TO_DELIVERY), kept
beside the recording's other stage files as audioset.jsonl. The export decides which become tags, at its thresholds
(melati/stage_export.py `audioset_thresholds`, `audioset_delivery`). It reads the vocal stem (speech and effects),
each segment on its own, as the frame head does. A recording already done is skipped. RUNS ON THE BOX, in
.venv-sed (deploy/box_sed_setup.sh).

    CUDA_VISIBLE_DEVICES=0 python nonverbal/audioset_pass.py --work $MELATI_DATA/podcast/work --slice 0/4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audioset_eval import SR, TAGS, Tagger, class_index, tag_probs, to_10fps  # noqa: E402

MODEL = "BEATs+ATST-F+M2D"
# the export's vocabulary (melati.nonverbal) -> audioset_eval.TAGS
CLASSES = {"laugh": "laughs", "sigh": "sighs", "gasp": "gasps", "cough": "coughs", "sneeze": "sneezes", "sniff": "sniffs",
           "throat_clear": "clears throat", "breath": "breathes", "cry": "cries", "yawn": "yawns", "scream": "screams",
           "groan": "groans", "hum": "hums", "whisper": "whisper", "shout": "shout", "sing": "sings"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--slice", default="0/1", help="i/n: every n-th finished recording from the i-th")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    i, n = map(int, a.slice.split("/"))
    names = class_index()
    idx = {c: [names[x] for x in TAGS[t]] for c, t in CLASSES.items()}
    recs = sorted(p.parent for p in a.work.glob("*/fill.jsonl"))[i::n]
    todo = [d for d in recs if not (d / "audioset.jsonl").exists()]
    print(f"[audioset] slice {a.slice}: {len(recs)} finished recordings, {len(todo)} to tag", flush=True)
    tagger = Tagger(MODEL, batch=a.batch)
    t0, secs = time.time(), 0.0
    for k, d in enumerate(todo, 1):
        segs = [json.loads(line) for line in (d / "segments.jsonl").open()]
        x, sr = sf.read(str(d / "vocal16k.flac"), dtype="float32")
        assert sr == SR, sr
        clips = [x[int(s["start"] * SR):int(s["end"] * SR)] for s in segs]
        rows = []
        for j in range(0, len(clips), 256):
            for s, c, p in zip(segs[j:j + 256], clips[j:j + 256], tagger(clips[j:j + 256])):
                pr = tag_probs(p, idx)
                rows.append({"seg": s["seg"], "model": MODEL, "fps": 10,
                             **{cls: [round(float(v), 3) for v in to_10fps(pr[cls])] for cls in CLASSES}})
        tmp = d / "audioset.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(d / "audioset.jsonl")
        secs += sum(len(c) for c in clips) / SR
        if k % 25 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"[audioset] {k}/{len(todo)} recordings, {secs / 3600:.1f} h of segments, {secs / el:.0f}x realtime",
                  flush=True)
    print("AUDIOSET_PASS_DONE", flush=True)


if __name__ == "__main__":
    main()
