#!/usr/bin/env python3
"""Laughs (and coughs) frame by frame, including over speech, where no gap finder can reach. RUNS ON THE BOX.

The gap finder (melati.nv_gaps) only looks between aligned words, so a laugh over speech (a speech-laugh, or one
person laughing while another talks) never becomes a candidate: 12 of NonverbalTTS's 56 test laughs have no gap at
all. This model reads a whole segment instead: the frozen openai/whisper-large-v3 encoder, eight of its layers
averaged to 10 frames a second, a learned mix of them, and a small dilated convolution over time (6 s of context)
that says, every 0.1 s, whether there is laughter, a cough, another vocal sound, or speech. Several can hold at once.

Trained on AMI (nonverbal/ami.py; CC BY 4.0): the only corpus at hand that marks every word and every laugh of a
whole conversation, so a frame with no mark is a negative rather than unknown. A vocal sound the aligner gave no
length is masked for a second around its position. With --nvtts, also on NonverbalTTS's training clips outside the
800 that choose thresholds (its test clips never): a clip with no laugh tag is laugh-free in every frame, and a laugh
tag, whose extent nobody marked, asks only that SOME frame within half a second of its word boundary fire (a
multiple-instance loss on the maximum). AMI alone transfers badly to NonverbalTTS's interviews: its clips without a
laugh were flagged nearly as often as those with one.

  extract   features and frame labels for AMI's meetings and NonverbalTTS's calibration and test clips (--feats,
            default /dev/shm: ~80 GB, rebuilt in minutes)
  train     the frame head on AMI train; the epoch kept is the best laugh average precision on AMI dev
  eval      events (frames over a threshold, joined across 0.1 s, at least 0.2 s) scored on AMI test against the
            marks, and on NonverbalTTS test as tags by the word slot they start in, alone and beside the gap head;
            thresholds chosen on AMI dev / NonverbalTTS's calibration clips for the target precision

    python nonverbal/frames.py extract
    python nonverbal/frames.py train --out $MELATI_DATA/models/nv_frames/whisper.pt
    python nonverbal/frames.py eval --model $MELATI_DATA/models/nv_frames/whisper.pt \\
        --gap-heads $MELATI_DATA/models/nv_heads_rel3 --gap-thresholds .../gap_eval_rel3.json --out ...
"""
import argparse, json, os, sys, zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from melati import nv_gaps as NG  # noqa: E402

from melati.nv_frames import CLASSES, FPS, LAYERS, SR, T, WIN_S, encoder, events, frame_head  # noqa: E402,F401

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh
TO_FRAME = {"laugh": "laugh", "cough": "cough", "speech": "speech", "filler": "speech"}   # else: other
POINT_MASK_S = 0.5
FEATS = Path("/dev/shm/melati_frames")


# ---------------------------------------------------------------- features
def grid(spans, n, pad=0.0):
    """Which 10 ms bins of n frames' worth the spans [{s, e}] cover (pad seconds either side)."""
    g = np.zeros(n * 10, bool)
    for s in spans:
        a, b = max(0, int((s["s"] - pad) * 100)), min(len(g), int(np.ceil((s["e"] + pad) * 100)))
        g[a:b] = True
    return g


def frame_labels(words, sounds, n):
    """(n, len(CLASSES)) int8 for n frames from t = 0: 1 present, 0 absent, -1 unknown. A word covering half a frame
    makes it speech; a timed mark covering 30 ms makes it that class; an untimed mark masks its class within
    POINT_MASK_S, since where the sound starts and ends is unknown."""
    lab = np.zeros((n, len(CLASSES)), np.int8)
    cov = grid(words, n).reshape(n, 10).mean(1)
    lab[:, CLASSES.index("speech")] = cov >= 0.5
    for k, c in enumerate(CLASSES[:3]):
        mine = [v for v in sounds if TO_FRAME.get(v["label"], "other") == c]
        timed = grid([v for v in mine if v["e"] > v["s"]], n).reshape(n, 10).mean(1) >= 0.3
        point = grid([v for v in mine if v["e"] <= v["s"]], n, POINT_MASK_S).reshape(n, 10).any(1)
        lab[:, k] = np.where(timed, 1, np.where(point, -1, 0))
    return lab


def extract_ami(run, root, out, batch):
    import soundfile as sf
    from ami import ANNOTATIONS, meetings, read, split_of, wav_path
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / ANNOTATIONS) as z:
        for m in meetings(z):
            if (out / f"{m}.json").exists() or not wav_path(root, m).exists():
                continue
            words, sounds = read(z, m)
            x, sr = sf.read(str(wav_path(root, m)), dtype="float32", always_2d=True)
            x = x.mean(1)
            assert sr == SR, sr
            wins = [x[int(i * WIN_S * SR):int((i + 1) * WIN_S * SR)] for i in range(int(np.ceil(len(x) / SR / WIN_S)))]
            wins = [w for w in wins if len(w) >= SR // 10]
            F, N = [], []
            for s in range(0, len(wins), batch):
                f, n = run(wins[s:s + batch])
                F.append(f); N += n
            F = np.concatenate(F)
            lab = frame_labels(words, sounds, len(wins) * T).reshape(len(wins), T, len(CLASSES))
            for i, n in enumerate(N):
                lab[i, n:] = -1                                            # past the end of the meeting
            np.save(out / f"{m}.npy", F)
            np.save(out / f"{m}.lab.npy", lab)
            (out / f"{m}.json").write_text(json.dumps({"meeting": m, "split": split_of(m), "windows": len(wins),
                                                       "frames": N}))
            print(f"  {m} ({split_of(m)}): {len(wins)} windows, laugh frames {int((lab[..., 0] == 1).sum())}",
                  flush=True)


PF_FRAME = {"Laughter": "laugh", "Breath": "other", "Words": "speech", "Repetitions": "speech", "Like": "speech",
            "You know": "speech", "Uh": "speech", "Um": "speech", "Agree": "speech", "Music": None, "Noise": None}


def pf_labels(events, words, n):
    """(n, len(CLASSES)) for one PodcastFillers episode. Its annotators labelled only what its detector proposed, so a
    class is known only inside a labelled event: 1 where the event is that class, 0 where it is another; elsewhere
    -1 (an Overlap or Other event says nothing either way). Speech is known everywhere from the episode's
    transcript (half a frame covered by a word)."""
    lab = np.full((n, len(CLASSES)), -1, np.int8)
    for k, c in enumerate(CLASSES[:3]):
        for kind, sel in ((0, [e for e in events if e[2] in PF_FRAME and PF_FRAME[e[2]] != c]),
                          (1, [e for e in events if e[2] in PF_FRAME and PF_FRAME[e[2]] == c])):
            hit = grid([{"s": a, "e": b} for a, b, _ in sel], n).reshape(n, 10).mean(1) >= 0.3
            lab[hit, k] = kind                    # positives last, so an overlap of the two reads as the class
    lab[:, CLASSES.index("speech")] = grid(words, n).reshape(n, 10).mean(1) >= 0.5
    return lab


def extract_pf(run, root, out, batch, licence):
    """PodcastFillers' episodes in 30 s windows: train and validation only where the audio is CC BY 3.0 (the
    release's rule, scripts/pf_licences.py), every test episode (they only score)."""
    import csv
    import soundfile as sf
    from build_manifest import resample
    from candidates import pf_words
    lic = json.loads(licence.read_text())
    out.mkdir(parents=True, exist_ok=True)
    events = {}
    with next(root.rglob("PodcastFillers.csv")).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            events.setdefault((r["episode_split_subset"], r["podcast_filename"]), []).append(
                (float(r["event_start_inepisode"]), float(r["event_end_inepisode"]), r["label_full_vocab"]))
    wav_root, tr_root = next(root.rglob("episode_wav")), next(root.rglob("episode_transcripts"))
    for (split, ep), evs in sorted(events.items()):
        name = f"{split}__{ep}"
        if (out / f"{name}.json").exists() or (split != "test" and lic.get(ep) != "CC BY 3.0"):
            continue
        wav, tr = wav_root / split / f"{ep}.wav", tr_root / split / f"{ep}.json"
        if not (wav.exists() and tr.exists()):
            continue
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = resample(x.mean(1), sr)
        wins = [x[int(i * WIN_S * SR):int((i + 1) * WIN_S * SR)] for i in range(int(np.ceil(len(x) / SR / WIN_S)))]
        wins = [w for w in wins if len(w) >= SR // 10]
        F, N = [], []
        for s0 in range(0, len(wins), batch):
            f, n = run(wins[s0:s0 + batch])
            F.append(f); N += n
        lab = pf_labels(evs, pf_words(tr), len(wins) * T).reshape(len(wins), T, len(CLASSES))
        for i, n in enumerate(N):
            lab[i, n:] = -1
        np.save(out / f"{name}.npy", np.concatenate(F))
        np.save(out / f"{name}.lab.npy", lab)
        (out / f"{name}.json").write_text(json.dumps({"meeting": name, "split": split, "windows": len(wins),
                                                     "frames": N, "episode": ep, "events": evs}))
        print(f"  pf {name}: {len(wins)} windows, laugh frames {int((lab[..., 0] == 1).sum())}", flush=True)


def extract_coughs(run, manifests, out, batch, seed=0):
    """COUGHVID's and Coswara's cough clips (nonverbal/build_manifest.py: each trimmed to its sound, at most 2.5 s, and
    padded with 0.3 s of silence) packed end to end into 30 s windows, per split: a frame is a cough inside a clip's
    sound and not one in its padding. Laughter and speech are absent throughout; another vocal sound is unknown."""
    import random
    import soundfile as sf
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for m in manifests for l in open(m, encoding="utf-8")]
    for split in ("train", "validation", "test"):
        name = f"coughs__{split}"
        if (out / f"{name}.json").exists():
            continue
        rs = [r for r in rows if r["split"] == split and Path(r["src"]["file"]).exists()]
        random.Random(seed).shuffle(rs)
        wins, labs, cur, lab, t = [], [], [], np.full((T, len(CLASSES)), -1, np.int8), 0.0
        for r in rs:
            x, _ = sf.read(r["src"]["file"], dtype="float32")
            if t + len(x) / SR > WIN_S:
                wins.append(np.concatenate(cur)); labs.append(lab)
                cur, lab, t = [], np.full((T, len(CLASSES)), -1, np.int8), 0.0
            n0, n1 = int(t * FPS), min(T, int(np.ceil((t + len(x) / SR) * FPS)))
            lab[n0:n1, [0, 1, 3]] = 0                       # laugh, cough, speech: absent
            a, b = int((t + 0.3) * FPS), min(T, int(np.ceil((t + 0.3 + r["dur"]) * FPS)))
            lab[a:b, 1] = 1
            cur.append(x); t += len(x) / SR
        if cur:
            wins.append(np.concatenate(cur)); labs.append(lab)
        F, N = [], []
        for s0 in range(0, len(wins), batch):
            f, n = run(wins[s0:s0 + batch])
            F.append(f); N += n
        lab = np.stack(labs)
        for i, n in enumerate(N):
            lab[i, n:] = -1
        np.save(out / f"{name}.npy", np.concatenate(F))
        np.save(out / f"{name}.lab.npy", lab)
        (out / f"{name}.json").write_text(json.dumps({"meeting": name, "split": split, "windows": len(wins),
                                                     "frames": N, "clips": len(rs)}))
        print(f"  coughs {split}: {len(rs)} clips in {len(wins)} windows", flush=True)


def extract_nvtts(run, cache, out, batch):
    from gap_rule import load
    out.mkdir(parents=True, exist_ok=True)
    for which in ("calib", "test", "rest"):
        if (out / f"{which}.json").exists():
            continue
        clips = load(cache, which)
        F, meta = [], []
        for s in range(0, len(clips), batch):
            part = clips[s:s + batch]
            f, n = run([x[: int(WIN_S * SR)] for _, x in part])
            F.append(f)
            meta += [{"row": cl["row"], "frames": k, "words": [{"s": w["s"], "e": w["e"]} for w in cl["words"]],
                      "tags": cl["tags"]} for (cl, _), k in zip(part, n)]
        np.save(out / f"{which}.npy", np.concatenate(F))
        (out / f"{which}.json").write_text(json.dumps(meta))
        print(f"  nvtts {which}: {len(meta)} clips", flush=True)


def nvtts_targets(meta, n_frames=T, around_s=0.5):
    """Frame labels and bags for one NonverbalTTS clip: speech from its aligned words; laugh and cough 0 in every
    frame of a clip with no tag of that class, else unknown, with one bag per tag (frames within around_s of its
    word boundary, at least one of which must fire); `other` unknown."""
    lab = np.full((n_frames, len(CLASSES)), -1, np.int8)
    n = meta["frames"]
    words = meta["words"]
    lab[:n, CLASSES.index("speech")] = grid(words, n).reshape(n, 10).mean(1) >= 0.5 if n else 0
    bags = []
    for k, c in enumerate(CLASSES[:2]):
        slots = [sl for cl, sl in meta["tags"] if cl == c]
        if not slots:
            lab[:n, k] = 0
        for sl in slots:
            a = words[sl - 1]["e"] if sl > 0 and sl - 1 < len(words) else 0.0
            b = words[sl]["s"] if sl < len(words) else n / FPS
            fa, fb = max(0, int((a - around_s) * FPS)), min(n, int(np.ceil((b + around_s) * FPS)))
            if fb > fa:
                bags.append((k, fa, fb))
    return lab, bags


# ---------------------------------------------------------------- model
def load_ami(feats: Path, split: str):
    """Every window of one split, memory-mapped: (features, labels)."""
    F, L = [], []
    for j in sorted(feats.glob("*.json")):
        meta = json.loads(j.read_text())
        if meta["split"] == split:
            F.append(np.load(feats / f"{meta['meeting']}.npy", mmap_mode="r"))
            L.append(np.load(feats / f"{meta['meeting']}.lab.npy"))
    return F, L


def predict(head, F, device="cuda", bs=32):
    """Per-frame probabilities for a list of (windows, T, L, D) arrays -> list of (windows, T, n)."""
    import torch
    out = []
    head.eval()
    with torch.no_grad():
        for f in F:
            ps = [torch.sigmoid(head(torch.from_numpy(np.asarray(f[s:s + bs])).to(device).float())).cpu().numpy()
                  for s in range(0, len(f), bs)]
            out.append(np.concatenate(ps))
    return out


def frame_ap(P, L, k):
    from sklearn.metrics import average_precision_score
    p = np.concatenate([x[..., k].ravel() for x in P])
    y = np.concatenate([x[..., k].ravel() for x in L])
    m = y >= 0
    return float(average_precision_score(y[m] == 1, p[m])) if (y[m] == 1).any() else None


def train(args):
    import torch
    F, L = load_ami(args.feats / "ami", "train")
    Fv, Lv = load_ami(args.feats / "ami", "validation")
    idx = [(i, j) for i, f in enumerate(F) for j in range(len(f))]
    bags = {}
    if args.pf:                         # PodcastFillers' CC BY training episodes, beside AMI's meetings
        Fp, Lp = load_ami(args.feats / "pf", "train")
        F += Fp; L += Lp
        n0 = len(idx)
        idx += [(i, j) for i, f in enumerate(F) if i >= len(F) - len(Fp) for j in range(len(f))
                if (L[i][j][:, :3] >= 0).any()]          # only windows holding a labelled event
        print(f"PodcastFillers: {len(Fp)} episodes, {len(idx) - n0} windows with a labelled event", flush=True)
    if args.coughs is not None:         # COUGHVID's and Coswara's coughs: about a tenth of the windows drawn
        Fc, Lc = load_ami(args.feats / "coughs", "train")
        F += Fc; L += Lc
        cw = [(len(F) - 1, j) for j in range(len(Fc[0]))] if Fc else []
        reps = max(1, int(0.1 * len(idx) / max(len(cw), 1)))
        idx += cw * reps
        print(f"coughs: {len(cw)} windows, each drawn {reps} times an epoch", flush=True)
    if args.nvtts:                      # NonverbalTTS's training clips: one more "meeting" of single windows
        meta = json.loads((args.feats / "nvtts" / "rest.json").read_text())
        tg = [nvtts_targets(m) for m in meta]
        F.append(np.load(args.feats / "nvtts" / "rest.npy", mmap_mode="r"))
        L.append(np.stack([t[0] for t in tg]))
        i = len(F) - 1
        nv_idx = [(i, j) for j in range(len(meta))]
        bags = {(i, j): t[1] for j, t in enumerate(tg)}
        print(f"NonverbalTTS rest: {len(meta)} clips, {sum(map(len, bags.values()))} tag bags", flush=True)
        idx += nv_idx * max(1, len(idx) // max(len(nv_idx), 1) // 2)      # about a third of the windows drawn
    lab = np.concatenate(L)
    pos = [(lab[..., k] == 1).sum() for k in range(len(CLASSES))]
    neg = [(lab[..., k] == 0).sum() for k in range(len(CLASSES))]
    pw = torch.tensor([min(20.0, float(np.sqrt(n / max(p, 1)))) for p, n in zip(pos, neg)], device="cuda")
    print(f"{len(idx)} train windows; positive frames {dict(zip(CLASSES, map(int, pos)))}; pos_weight "
          f"{[round(float(v), 2) for v in pw]}", flush=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    head = frame_head().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    per = len(idx) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * per, pct_start=0.1)
    best, curve = None, []
    for ep in range(args.epochs):
        head.train()
        order = rng.permutation(len(idx))
        for b in range(per):
            sel = [idx[q] for q in order[b * args.bs:(b + 1) * args.bs]]
            x = torch.from_numpy(np.stack([F[i][j] for i, j in sel])).cuda().float()
            y = torch.from_numpy(np.stack([L[i][j] for i, j in sel])).cuda().float()
            logit = head(x)
            m = (y >= 0).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y.clamp(min=0), reduction="none",
                                                                       pos_weight=pw)
            loss = (loss * m).sum() / m.sum().clamp(min=1)
            mil = [logit[q, fa:fb, k].max() for q, ij in enumerate(sel) for k, fa, fb in bags.get(ij, ())]
            if mil:                         # some frame of each tagged region is that class
                z = torch.stack(mil)
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(z, torch.ones_like(z))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        P = predict(head, Fv)
        aps = {c: frame_ap(P, Lv, k) for k, c in enumerate(CLASSES)}
        curve.append({"epoch": ep + 1, **{c: round(v, 4) for c, v in aps.items() if v is not None}})
        print(f"  epoch {ep + 1}: dev frame AP {curve[-1]}", flush=True)
        if best is None or aps["laugh"] > best[0]:
            best = (aps["laugh"], {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classes": list(CLASSES), "layers": list(LAYERS), "fps": FPS, "state": best[1], "curve": curve,
                "trained_on": "AMI train meetings (CC BY 4.0)"}, args.out)
    print(f"TRAIN_DONE best dev laugh AP {best[0]:.4f} -> {args.out}", flush=True)


# ---------------------------------------------------------------- events and scoring
def ami_marks(root: Path, meeting: str, z):
    from ami import read, union
    import bisect
    words, sounds = read(z, meeting)
    spans = union(words)
    starts = [s["s"] for s in spans]
    marks = []
    for v in sounds:
        t = (v["s"] + max(v["e"], v["s"])) / 2
        i = bisect.bisect_right(starts, t) - 1
        marks.append({"cls": TO_FRAME.get(v["label"], "other"), "s": v["s"], "e": max(v["e"], v["s"]),
                      "point": v["e"] <= v["s"], "in_speech": bool(i >= 0 and spans[i]["e"] > t)})
    return marks


def near(ev, m):
    tol = POINT_MASK_S if m["point"] else 0.1
    return ev[0] <= m["e"] + tol and ev[1] >= m["s"] - tol


def score_ami(data, cls, t):
    """Events of one class at threshold t over meetings [(probs per frame, marks)] -> precision, recall."""
    r = {"events": 0, "right": 0, "marks": 0, "found": 0, "marks_in_speech": 0, "found_in_speech": 0}
    for p, marks in data:
        ev = [(a / FPS, b / FPS) for a, b, _ in events(p, t)]
        mk = [m for m in marks if m["cls"] == cls]
        r["events"] += len(ev)
        r["right"] += sum(any(near(e, m) for m in mk) for e in ev)
        for m in mk:
            hit = any(near(e, m) for e in ev)
            r["marks"] += 1; r["found"] += hit
            r["marks_in_speech"] += m["in_speech"]; r["found_in_speech"] += hit and m["in_speech"]
    r["precision"] = round(r["right"] / r["events"], 4) if r["events"] else None
    r["recall"] = round(r["found"] / r["marks"], 4) if r["marks"] else None
    r["recall_in_speech"] = round(r["found_in_speech"] / r["marks_in_speech"], 4) if r["marks_in_speech"] else None
    return r


GRID = [round(float(t), 2) for t in np.arange(0.05, 0.96, 0.05)] + [0.97, 0.98, 0.99, 0.995, 0.998, 0.999]


def lowest(score_at, target, min_hyp=5, key="events"):
    """The lowest threshold whose precision reaches target, on the gap thresholds' 0.05 grid and past 0.95 (a
    sigmoid output trained against confident frame labels sits near 1 where it is sure)."""
    for t in GRID:
        r = score_at(t)
        if r[key] < min_hyp:
            return None
        if r["precision"] is not None and r["precision"] >= target:
            return t
    return None


def score_pf(data, t, want="Laughter"):
    """Events at threshold t over PodcastFillers episodes [(probs per frame, events)]: hand-timed events of `want`
    found, and the events' precision where PodcastFillers has a label (one overlapping no label is unscored)."""
    r = {"events": 0, "right": 0, "wrong": 0, "refs": 0, "found": 0}
    for p, evs in data:
        ev = [(a / FPS, b / FPS) for a, b, _ in events(p, t)]
        lab = [(s_, e_, l) for s_, e_, l in evs if l in PF_FRAME]
        refs = [(s_, e_) for s_, e_, l in evs if l == want]
        r["events"] += len(ev)
        for a, b in ev:
            ov = {l for s_, e_, l in lab if s_ <= b and e_ >= a}
            r["right"] += want in ov
            r["wrong"] += bool(ov) and want not in ov
        r["refs"] += len(refs)
        r["found"] += sum(any(a <= e_ and b >= s_ for a, b in ev) for s_, e_ in refs)
    n = r["right"] + r["wrong"]
    r["labelled"] = n
    r["precision"] = round(r["right"] / n, 4) if n else None
    r["recall"] = round(r["found"] / r["refs"], 4) if r["refs"] else None
    return r


def slot_of(a_s, words):
    """The word slot an event starting at a_s seconds is tagged at: before the first word whose middle is after it."""
    return sum((w["s"] + w["e"]) / 2 < a_s for w in words)


def nvtts_tags(p, words, t, over_speech_only=False):
    out = []
    for a, b, peak in events(p, t):
        a_s, b_s = a / FPS, b / FPS
        if over_speech_only:
            cov = sum(max(0.0, min(b_s, w["e"]) - max(a_s, w["s"])) for w in words)
            if cov < 0.5 * (b_s - a_s):
                continue
        out.append(slot_of(a_s, words))
    return out


def score_slots(clips, hyps, cls="laugh", tol=1):
    """Tag precision and recall of slot hypotheses against NonverbalTTS's reference tags (eval_nonverbal_gaps.score's
    rule: right within `tol` words)."""
    s = {"ref": 0, "found": 0, "hyp": 0, "right": 0}
    for cl, hyp in zip(clips, hyps):
        ref = [k for c, k in cl["tags"] if c == cls]
        hyp = sorted(set(hyp))
        s["ref"] += len(ref); s["hyp"] += len(hyp)
        s["found"] += sum(any(abs(r - h) <= tol for h in hyp) for r in ref)
        s["right"] += sum(any(abs(h - r) <= tol for r in ref) for h in hyp)
    s["precision"] = round(s["right"] / s["hyp"], 4) if s["hyp"] else None
    s["recall"] = round(s["found"] / s["ref"], 4) if s["ref"] else None
    return s


def gap_tags(clips_x, heads, thresholds, classes=("laugh", "cough"), batch=64):
    """The gap head's tags per class and clip, as the pipeline makes them -> {class: [[slot, ...] per clip]}."""
    from melati.ser import HeadNV
    head = HeadNV("whisper", heads_dir=str(heads))
    cands, clips = [], []
    for i, (cl, x) in enumerate(clips_x):
        for c in NG.candidates(x, cl["words"], floor_db=2.0):
            cands.append((i, c["slot"]))
            clips.append(NG.clip_of(x, c))
    P = np.concatenate([head(clips[s:s + batch])[0] for s in range(0, len(clips), batch)])
    out = {c: [[] for _ in clips_x] for c in classes}
    for (i, slot), row in zip(cands, P):
        c = NG.decide(dict(zip(NG.HEAD_LABELS, map(float, row))), thresholds)
        if c in out:
            out[c][i].append(slot)
    return out


def evaluate(args):
    import torch
    from ami import ANNOTATIONS
    from gap_rule import load
    ck = torch.load(args.model, map_location="cpu")
    head = frame_head().cuda()
    head.load_state_dict(ck["state"])
    res = {"model": str(args.model), "target_precision": args.target}
    # AMI: dev chooses the threshold, test scores it
    data = {}
    with zipfile.ZipFile(args.ami / ANNOTATIONS) as z:
        for sp in ("validation", "test"):
            F, _ = load_ami(args.feats / "ami", sp)
            metas = [json.loads(j.read_text()) for j in sorted((args.feats / "ami").glob("*.json"))]
            metas = [m for m in metas if m["split"] == sp]
            P = predict(head, F)
            data[sp] = [(np.concatenate([w[:n] for w, n in zip(p, m["frames"])]), ami_marks(args.ami, m["meeting"], z))
                        for p, m in zip(P, metas)]
    res["ami"], ami_test = {}, {}
    for k, cls in enumerate(("laugh", "cough")):
        dev = [(p[:, k], mk) for p, mk in data["validation"]]
        test = ami_test[cls] = [(p[:, k], mk) for p, mk in data["test"]]
        t = lowest(lambda t: score_ami(dev, cls, t), args.target)
        res["ami"][cls] = {"threshold": t, "dev": score_ami(dev, cls, t) if t else None,
                           "test": score_ami(test, cls, t) if t else None}
        print("AMI", cls, json.dumps(res["ami"][cls]), flush=True)
    # PodcastFillers: its CC BY validation episodes choose the threshold, its test episodes score it
    if (args.feats / "pf").exists():
        pfd = {}
        for sp in ("validation", "test"):
            F, _ = load_ami(args.feats / "pf", sp)
            metas = [m for m in (json.loads(j.read_text()) for j in sorted((args.feats / "pf").glob("*.json")))
                     if m["split"] == sp]
            pfd[sp] = [(np.concatenate([w[:n] for w, n in zip(p, m["frames"])])[:, 0], [tuple(e) for e in m["events"]])
                       for p, m in zip(predict(head, F), metas)]
        t = lowest(lambda t: score_pf(pfd["validation"], t), args.target, key="labelled")
        res["podcastfillers"] = {"threshold": t, "validation": score_pf(pfd["validation"], t) if t else None,
                                 "test": score_pf(pfd["test"], t) if t else None,
                                 "test_at_ami_threshold": score_pf(pfd["test"], res["ami"]["laugh"]["threshold"])
                                 if res["ami"]["laugh"]["threshold"] else None}
        if t:                              # the laugh threshold that ships is this one: AMI at it too
            res["ami"]["laugh"]["test_at_pf_threshold"] = score_ami(ami_test["laugh"], "laugh", t)
        print("PodcastFillers laugh", json.dumps(res["podcastfillers"]), flush=True)
    # NonverbalTTS: its calibration clips choose the threshold, its test clips score it; alone and beside the gap head
    meta = {w: json.loads((args.feats / "nvtts" / f"{w}.json").read_text()) for w in ("calib", "test")}
    P = {w: [p[:m["frames"]] for p, m in zip(predict(head, [np.load(args.feats / "nvtts" / f"{w}.npy",
                                                                    mmap_mode="r")])[0], meta[w])]
         for w in ("calib", "test")}
    ev = json.loads(args.gap_thresholds.read_text())
    gthr = {c: v["threshold"] for c, v in ev[args.rule]["thresholds"].items() if v}
    gap = {w: gap_tags(load(args.cache, w), args.gap_heads, gthr) for w in ("calib", "test")}
    res["nonverbaltts"] = {"gap_thresholds": gthr}
    for k, cls in enumerate(("laugh", "cough")):
        r = res["nonverbaltts"][cls] = {"gap_head": {w: score_slots(meta[w], gap[w][cls], cls) for w in gap}}
        for mode in ("alone", "over_speech", "union"):
            def hyps(w, t):
                fr = [nvtts_tags(p[:, k], m["words"], t, over_speech_only=(mode == "over_speech"))
                      for p, m in zip(P[w], meta[w])]
                return fr if mode == "alone" else [g + f for g, f in zip(gap[w][cls], fr)]
            t = lowest(lambda t: score_slots(meta["calib"], hyps("calib", t), cls), args.target, key="hyp")
            r[mode] = {"threshold": t, "calib": score_slots(meta["calib"], hyps("calib", t), cls) if t else None,
                       "test": score_slots(meta["test"], hyps("test", t), cls) if t else None}
            print("NonverbalTTS", cls, mode, json.dumps(r[mode]), flush=True)
        print("NonverbalTTS", cls, "gap head", json.dumps(r["gap_head"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("FRAMES_EVAL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "train", "eval"])
    ap.add_argument("--ami", type=Path, default=Path(f"{MELATI_DATA}/nv/raw/ami"))
    ap.add_argument("--cache", type=Path, default=Path(f"{MELATI_DATA}/nv/nvtts_align.json"))
    ap.add_argument("--feats", type=Path, default=FEATS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", type=Path, default=Path(f"{MELATI_DATA}/models/nv_frames/whisper.pt"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gap-heads", type=Path, default=Path(f"{MELATI_DATA}/models/nv_heads"))
    ap.add_argument("--gap-thresholds", type=Path, default=None)
    ap.add_argument("--rule", default='{"floor_db": 2}')
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--nvtts", action="store_true", help="train: also on NonverbalTTS's training clips")
    ap.add_argument("--pf", action="store_true", help="extract / train: PodcastFillers' episodes instead of NonverbalTTS")
    ap.add_argument("--coughs", nargs="*", type=Path, default=None,
                    help="extract: these cough manifests (build_manifest.py's); train: with no value, use them")
    ap.add_argument("--podcastfillers", type=Path, default=Path(f"{MELATI_DATA}/bench/podcastfillers"))
    ap.add_argument("--pf-licence", type=Path, default=Path(f"{MELATI_DATA}/nv/pf_episode_licence.json"))
    args = ap.parse_args()
    if args.step == "extract":
        run = encoder()
        extract_ami(run, args.ami, args.feats / "ami", args.batch)
        if args.coughs:                 # manifests given: build the cough windows
            extract_coughs(run, args.coughs, args.feats / "coughs", args.batch)
        elif args.pf:
            extract_pf(run, args.podcastfillers, args.feats / "pf", args.batch, args.pf_licence)
        else:
            extract_nvtts(run, args.cache, args.feats / "nvtts", args.batch)
        print("EXTRACT_DONE", flush=True)
    elif args.step == "train":
        args.out = args.out or args.model
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
