#!/usr/bin/env python3
"""AudioSet's frame-level taggers as a source of the tags Melati does not have yet, class by class. PretrainedSED
(Schmid et al., ICASSP 2025; MIT) fine-tuned BEATs, ATST-F, M2D, ASiT and fPaSST on AudioSet's strong labels: 447
classes, a probability every 40 ms. Each candidate tag (TAGS: its AudioSet classes, the most probable of them per
frame) is scored wherever labels exist, at every threshold of nonverbal/frames.py's grid. RUNS ON THE BOX, in
.venv-sed (deploy/box_sed_setup.sh).

  vocalsound  VocalSound's test speakers (one sound a clip: laughter, sigh, cough, throat clearing, sneeze, sniff;
              CC BY-SA 4.0, so it only scores): of each sound's clips, the share where a tag fires
  expresso    Expresso's read utterances by style (CC BY-NC 4.0, scoring only): of each style's utterances, the
              share where a tag covers at least half the frames (whispering is a delivery, not an event)
  podcasts    PodcastFillers' test episodes, whole (English podcasts nothing here trained on): each tag's events an
              hour, what PodcastFillers labelled where they land, and laughter and breath scored as the frame head
              was (frames.score_pf, at its 10 frames a second)
  ami         AMI's test meetings: laughs and coughs scored as the frame head was (frames.score_ami)
  nvtts       NonverbalTTS, every aligned clip (interviews and expressive speech, CC BY-NC-SA 4.0, scoring only):
              laughs, breaths, coughs, throat clearing, sniffs, sneezes and groans, each tag right within one word

    python nonverbal/audioset_eval.py --model BEATs --out $MELATI_DATA/results/nonverbal/audioset_BEATs.json
"""
import os
import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh
MELATI_ROOT = os.environ.get("MELATI_ROOT", "/root/melati")        # environments, checkouts: deploy/env.sh

ROOT = Path(__file__).resolve().parents[1]
PSED = Path(f"{MELATI_ROOT}/ext/PretrainedSED")
sys.path[:0] = [str(ROOT), str(ROOT / "nonverbal"), str(PSED)]
from frames import GRID, ami_marks, score_ami, score_pf  # noqa: E402
from melati.nv_frames import events  # noqa: E402

SR, FPS_IN, CHUNK_S = 16000, 25, 10      # every PretrainedSED model: 16 kHz, 10 s windows, 250 frames each
# a Melati tag and the AudioSet classes that are it; a name missing from the strong label set is dropped (and listed)
TAGS = {
    "laughs": ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "sighs": ["Sigh"],
    "gasps": ["Gasp"],
    "coughs": ["Cough"],
    "sneezes": ["Sneeze"],
    "sniffs": ["Sniff"],
    "clears throat": ["Throat clearing"],
    "breathes": ["Breathing"],
    "cries": ["Crying, sobbing", "Whimper"],
    "hums": ["Humming"],
    "screams": ["Screaming"],
    "groans": ["Groan"],
    "grunts": ["Grunt"],
    "snorts": ["Snort"],
    "yawns": ["Yawn"],
    "whisper": ["Whispering"],
    "shout": ["Shout", "Yell", "Bellow"],
    "sings": ["Singing"],
}


def load_model(name):
    """PretrainedSED's model with its strong-label checkpoint, built from inside its folder: it keeps checkpoints at
    the relative path resources/, and downloads one there on first use."""
    import os
    here = os.getcwd()
    os.chdir(PSED)
    try:
        return _load_model(name)
    finally:
        os.chdir(here)


def _load_model(name):
    from models.prediction_wrapper import PredictionsWrapper
    if name == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        m = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    elif name == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        m = PredictionsWrapper(ATSTWrapper(), checkpoint="ATST-F_strong_1")
    elif name == "M2D":
        from models.m2d.M2D_wrapper import M2DWrapper
        w = M2DWrapper()
        m = PredictionsWrapper(w, checkpoint="M2D_strong_1", embed_dim=w.m2d.cfg.feature_d)
    else:
        raise SystemExit(f"no model {name}")
    return m.eval().cuda()


def class_index():
    """{AudioSet display name: index in the models' 447 outputs}."""
    from data_util import audioset_classes as ac
    return {name: i for i, name in enumerate(ac.as_strong_train_classes)}


class Tagger:
    """One model, or several joined by + (BEATs+ATST-F+M2D), whose probabilities are averaged frame by frame."""

    def __init__(self, name, batch=32):
        import torch
        self.torch, self.models, self.batch = torch, [load_model(n) for n in name.split("+")], batch

    def __call__(self, xs):
        """Clips (16 kHz) -> [per clip (447, frames) probabilities at 25 a second]."""
        torch = self.torch
        seg = SR * CHUNK_S
        pieces, owner = [], []
        for k, x in enumerate(xs):
            for i in range(max(1, int(np.ceil(len(x) / seg)))):
                c = x[i * seg:(i + 1) * seg]
                pieces.append(np.pad(c, (0, seg - len(c))).astype(np.float32))
                owner.append(k)
        outs = []
        for i in range(0, len(pieces), self.batch):
            w = torch.from_numpy(np.stack(pieces[i:i + self.batch])).cuda()
            with torch.no_grad():
                ys = [torch.sigmoid(m(m.mel_forward(w))[0].float()) for m in self.models]
            outs.append(torch.stack(ys).mean(0).cpu().numpy())
        Y = np.concatenate(outs)                                     # (pieces, 447, 250)
        res = []
        for k, x in enumerate(xs):
            p = np.concatenate([Y[j] for j, o in enumerate(owner) if o == k], axis=1)
            res.append(p[:, :max(1, int(np.ceil(len(x) / SR * FPS_IN)))])
        return res


def tag_probs(p, idx):
    """(447, frames) -> {tag: per-frame probability, the most probable of its classes}."""
    return {t: p[ix].max(0) for t, ix in idx.items()}


def to_10fps(p):
    """25 frames a second -> 10, each 0.1 s the most probable of the frames it overlaps (frames.py's scorers)."""
    n = int(np.ceil(len(p) / 2.5))
    lo = np.floor(np.arange(n) * 2.5).astype(int)
    return np.maximum.reduceat(p, lo) if len(p) else p


def read16(src, sr_target=SR):
    import soundfile as sf
    import librosa
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=sr_target) if sr != sr_target else x


def vocalsound(tagger, idx, root):
    z = zipfile.ZipFile(root / "vs_release_44k.zip")
    files = {Path(n).name: n for n in z.namelist() if n.endswith(".wav") and "MACOSX" not in n}
    names = {r[1]: r[2] for r in csv.reader(io.TextIOWrapper(z.open("class_labels_indices_vs.csv")))}
    test = [(files[Path(e["wav"]).name], names[e["labels"]]) for e in json.load(z.open("datafiles/te_rev.json"))["data"]
            if Path(e["wav"]).name in files]
    scores = defaultdict(list)                                     # sound -> [ {tag: clip max} ]
    for i in range(0, len(test), 256):
        part = test[i:i + 256]
        xs = [read16(io.BytesIO(z.read(n))) for n, _ in part]
        for (_, sound), p in zip(part, tagger(xs)):
            scores[sound].append({t: float(v.max()) for t, v in tag_probs(p, idx).items()})
    out = {"clips": {s: len(v) for s, v in scores.items()}, "fires": {}}
    for t in idx:
        out["fires"][t] = {str(th): {s: round(float(np.mean([c[t] >= th for c in v])), 4) for s, v in scores.items()}
                           for th in GRID}
    return out


def expresso(tagger, idx, root):
    import pyarrow.parquet as pq
    cover = defaultdict(list)                                       # style -> [ {tag: per-frame probs} ]
    for f in sorted((root / "read").glob("*.parquet")):
        rows = pq.read_table(f, columns=["audio", "style"]).to_pylist()
        for i in range(0, len(rows), 128):
            part = rows[i:i + 128]
            xs = [read16(io.BytesIO(r["audio"]["bytes"])) for r in part]
            for r, p in zip(part, tagger(xs)):
                cover[r["style"]].append({t: tag_probs(p, idx)[t].astype(np.float16) for t in ("whisper", "laughs",
                                                                                             "shout", "sings")
                                          if t in idx})
    out = {"utterances": {s: len(v) for s, v in cover.items()}, "covers_half": {}}
    for t in ("whisper", "laughs", "shout", "sings"):
        if t in idx:
            out["covers_half"][t] = {str(th): {s: round(float(np.mean([np.mean(u[t] >= th) >= 0.5 for u in v])), 4)
                                               for s, v in cover.items()} for th in GRID}
    return out


def podcasts(tagger, idx, root):
    evs = defaultdict(list)
    with next(root.rglob("PodcastFillers.csv")).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["episode_split_subset"] == "test":
                evs[r["podcast_filename"]].append((float(r["event_start_inepisode"]),
                                                   float(r["event_end_inepisode"]), r["label_full_vocab"]))
    wav_root = next(root.rglob("episode_wav")) / "test"
    data, hours = [], 0.0
    for ep, ev in sorted(evs.items()):
        if not (wav_root / f"{ep}.wav").exists():
            continue
        x = read16(str(wav_root / f"{ep}.wav"))
        hours += len(x) / SR / 3600
        p = tagger([x])[0]
        data.append(({t: to_10fps(v) for t, v in tag_probs(p, idx).items()}, ev))
    out = {"episodes": len(data), "hours": round(hours, 2), "per_hour": {}, "lands_on": {}}
    for t in idx:
        out["per_hour"][t], out["lands_on"][t] = {}, {}
        for th in GRID:
            n, on = 0, Counter()
            for probs, ev in data:
                for a, b, _ in events(probs[t], th):
                    n += 1
                    a_s, b_s = a / 10, b / 10
                    labs = {l for s_, e_, l in ev if s_ <= b_s and e_ >= a_s}
                    on.update(labs or {"(no label)"})
            out["per_hour"][t][str(th)] = round(n / hours, 3)
            out["lands_on"][t][str(th)] = dict(on.most_common(8))
    for t, want in (("laughs", "Laughter"), ("breathes", "Breath")):
        if t in idx:
            out[want] = {str(th): score_pf([(probs[t], ev) for probs, ev in data], th, want=want) for th in GRID}
    return out


NVTTS = {"laughs": "laugh", "breathes": "breath", "coughs": "cough", "clears throat": "throat_clear", "sniffs": "sniff",
         "sneezes": "sneeze", "groans": "groan", "grunts": "groan"}


def nvtts(tagger, idx, cache):
    """NonverbalTTS, every aligned clip (none of it trained these models): a tag is placed at the word slot its event
    starts in and is right within one word of a reference tag of its class (frames.score_slots)."""
    from frames import slot_of, score_slots
    from gap_rule import load
    clips, probs = [], []
    for which in ("test", "calib", "train"):
        part = load(cache, which)
        for i in range(0, len(part), 64):
            chunk = part[i:i + 64]
            for (cl, _), p in zip(chunk, tagger([x for _, x in chunk])):
                clips.append(cl)
                probs.append({t: to_10fps(v) for t, v in tag_probs(p, idx).items() if t in NVTTS})
    out = {"clips": len(clips), "hours": round(sum(cl["dur"] for cl in clips) / 3600, 2),
           "refs": dict(Counter(c for cl in clips for c, _ in cl["tags"]))}
    # two halves of the clips, by a fixed draw: a threshold chosen on one is measured on the other
    half = np.random.RandomState(0).rand(len(clips)) < 0.5
    for t, cls in NVTTS.items():
        if t in idx:
            hyps = {th: [[slot_of(a / 10, cl["words"]) for a, _, _ in events(pr[t], th)] for cl, pr in zip(clips, probs)]
                    for th in GRID}
            out[t] = {str(th): score_slots(clips, h, cls) for th, h in hyps.items()}
            out[t + "/halves"] = [{str(th): score_slots([c for c, k in zip(clips, half) if k == side],
                                                       [x for x, k in zip(h, half) if k == side], cls)
                                   for th, h in hyps.items()} for side in (True, False)]
    return out


def ami(tagger, idx, root):
    from ami import ANNOTATIONS, meetings, split_of, wav_path
    data = []
    with zipfile.ZipFile(root / ANNOTATIONS) as z:
        for m in meetings(z):
            if split_of(m) != "test" or not wav_path(root, m).exists():
                continue
            x = read16(str(wav_path(root, m)))
            probs = {t: to_10fps(v) for t, v in tag_probs(tagger([x])[0], idx).items()}
            data.append((probs, ami_marks(root, m, z)))
    return {"meetings": len(data),
            **{cls: {str(th): score_ami([(p[t], mk) for p, mk in data], cls, th) for th in GRID}
               for t, cls in (("laughs", "laugh"), ("coughs", "cough")) if t in idx}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="BEATs", help="BEATs, ATST-F or M2D, or several joined by +")
    ap.add_argument("--sets", nargs="+", default=["vocalsound", "expresso", "podcasts", "ami"])
    ap.add_argument("--vocalsound", type=Path, default=Path(f"{MELATI_DATA}/nv/raw/vocalsound"))
    ap.add_argument("--expresso", type=Path, default=Path(f"{MELATI_DATA}/emo/raw/expresso"))
    ap.add_argument("--podcastfillers", type=Path, default=Path(f"{MELATI_DATA}/bench/podcastfillers"))
    ap.add_argument("--ami", type=Path, default=Path(f"{MELATI_DATA}/nv/raw/ami"))
    ap.add_argument("--nvtts-cache", type=Path, default=Path(f"{MELATI_DATA}/nv/nvtts_align.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = class_index()
    idx = {t: [names[c] for c in cs if c in names] for t, cs in TAGS.items()}
    res = json.loads(a.out.read_text()) if a.out.exists() else {}         # a rerun adds sets to what is there
    res.update(model=a.model, classes={t: [c for c in cs if c in names] for t, cs in TAGS.items()},
               missing=sorted(c for cs in TAGS.values() for c in cs if c not in names), grid=GRID)
    idx = {t: ix for t, ix in idx.items() if ix}
    print(f"{a.model}: {len(names)} classes; tags {sorted(idx)}; not in AudioSet strong {res['missing']}", flush=True)
    tagger = Tagger(a.model)
    for s in a.sets:
        res[s] = {"vocalsound": lambda: vocalsound(tagger, idx, a.vocalsound),
                  "expresso": lambda: expresso(tagger, idx, a.expresso),
                  "podcasts": lambda: podcasts(tagger, idx, a.podcastfillers),
                  "ami": lambda: ami(tagger, idx, a.ami),
                  "nvtts": lambda: nvtts(tagger, idx, a.nvtts_cache)}[s]()
        print(f"{s} done", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    print("AUDIOSET_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
