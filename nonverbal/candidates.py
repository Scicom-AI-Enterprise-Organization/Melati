#!/usr/bin/env python3
"""Training clips that ARE gap candidates: what the pipeline will actually ask the non-verbal heads about. ON THE BOX.

A head trained only on hand-cut events (nonverbal/build_manifest.py) classifies such events well (macro F1 0.85 on
held-out PodcastFillers and VocalSound) and real candidates badly: a candidate is whatever sound sits in a word gap,
cut by an energy rule, and most are word tails, clicks and breaths no one cut by hand. So candidates are found here
with the pipeline's own rule (melati.nv_gaps.candidates, clip_of), and labelled from what the corpus says is there:

  pf      PodcastFillers train / validation episodes, with the word times its own transcripts carry. A candidate
          is labelled by the hand-timed event it overlaps most (at least 30% of the candidate, or half the event):
          Uh / Um / Agree -> filler, Words / Repetitions -> speech, Breath, Laughter, Music / Noise -> noise.
          A candidate that overlaps no event is left out: PodcastFillers' annotators labelled only the regions its
          detector proposed, so an unlabelled one is unknown, not "nothing".
  nvtts   NonverbalTTS train clips OUTSIDE the 800 that choose the thresholds (scripts/eval_nonverbal_gaps.py
          draws them with seed 0), with the word times nonverbal/align_cache.py stored. A candidate at the boundary of a
          reference tag is that class; one with no tag within a word is `other` (a sound no annotator tagged: word
          tails, clicks, noise, and the breaths annotators skipped); one a word away from a tag is ambiguous and
          left out. NonverbalTTS's labels are CC BY-NC-SA 4.0, so these never train a released head.
  ami     AMI meetings (nonverbal/ami.py; CC BY 4.0), the headset mix, in the gaps between ANYONE's words. A candidate
          is labelled by the mark it overlaps most, as for PodcastFillers: a word is speech (um, uh, mm-hmm: filler), a
          vocal sound its class. A sound the aligner gave no length marks only its gap, so it labels the gap's
          candidate when there is exactly one. AMI marks no ordinary breathing, so an unmarked candidate is left
          out, unless --unmarked names a label for it: `other`, as NonverbalTTS's untagged candidates were.
          --words instead cuts words out by their own times, padded as a candidate is: every um / uh / mm-hmm as
          `filler`, and the second half of one word in --word-every as `speech`, the word tail a misaligned gap
          holds (PodcastFillers' word events did this for the heads that trained on it). All three splits are
          built: the test meetings score the heads (nonverbal/eval_ami.py).

Clips go to --clips (default /dev/shm, as build_manifest.py's), rows to --out (one manifest per source).

    python nonverbal/candidates.py pf --podcastfillers $MELATI_DATA/bench/podcastfillers \\
        --out $MELATI_DATA/nv/manifest_pfc.jsonl
    python nonverbal/candidates.py nvtts --cache $MELATI_DATA/nv/nvtts_align.json \\
        --out $MELATI_DATA/nv/manifest_nvc.jsonl
    python nonverbal/candidates.py ami --ami $MELATI_DATA/nv/raw/ami --out $MELATI_DATA/nv/manifest_amic.jsonl
"""
import argparse, bisect, csv, hashlib, json, os, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manifest import PF_MAP, resample, write  # noqa: E402
from melati.nv_gaps import SR, TO_CLASS, candidates, clip_of  # noqa: E402

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh

NAME_OF = {v: k for k, v in TO_CLASS.items()}          # NonverbalTTS class -> head label ("throat_clear" -> ...)


def pf_words(path: Path):
    """Word times from a PodcastFillers transcript (offsets and durations in 100 ns ticks)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    words = []
    for seg in d.get("segments", []):
        for w in (seg.get("nbest") or [{}])[0].get("words", []):
            s = w["offset"] / 1e7
            words.append({"w": w["text"], "s": s, "e": s + w["duration"] / 1e7})
    return sorted(words, key=lambda w: w["s"])


def windows(words, dur, max_s=28.0, pause_s=1.0, edge_s=0.5):
    """Runs of words like the pipeline's segments: broken at a pause longer than pause_s or past max_s, each
    reaching up to edge_s (and at most half the pause) into the silence on either side."""
    runs, cur = [], [words[0]]
    for w in words[1:]:
        if w["s"] - cur[-1]["e"] > pause_s or w["e"] - cur[0]["s"] > max_s:
            runs.append(cur); cur = [w]
        else:
            cur.append(w)
    runs.append(cur)
    out = []
    for k, r in enumerate(runs):
        before = r[0]["s"] - (runs[k - 1][-1]["e"] if k else 0.0)
        after = (runs[k + 1][0]["s"] if k + 1 < len(runs) else dur) - r[-1]["e"]
        a, b = max(0.0, r[0]["s"] - min(edge_s, before / 2)), min(dur, r[-1]["e"] + min(edge_s, after / 2))
        out.append((a, b, r))
    return out


def label_pf(s, e, events, starts, mapping=PF_MAP):
    """The mapped label of the event that overlaps [s, e] most, if the overlap is substantial (mapping=None: the
    events carry head labels already)."""
    best, best_ov = None, 0.0
    i = bisect.bisect_left(starts, s - 30.0)
    for es, ee, lab in events[i:]:
        if es > e:
            break
        ov = min(e, ee) - max(s, es)
        if ov > best_ov:
            best, best_ov = (es, ee, lab), ov
    if best is None:
        return None
    es, ee, lab = best
    if best_ov >= 0.3 * (e - s) or best_ov >= 0.5 * (ee - es):
        return lab if mapping is None else mapping.get(lab)
    return None


def label_any(s, e, events, starts):
    """Whether any event overlaps [s, e] at all."""
    i = bisect.bisect_left(starts, s - 30.0)
    for es, ee, _ in events[i:]:
        if es > e:
            return False
        if min(e, ee) > max(s, es):
            return True
    return False


def pf_episode(job):
    wav, transcript, events, split, out, rule, corpus = job
    words = pf_words(Path(transcript))
    if not words:
        return []
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = resample(x.mean(1), sr)
    dur = len(x) / SR
    events = sorted(events)
    starts = [ev[0] for ev in events]
    tag = hashlib.sha1(Path(wav).stem.encode()).hexdigest()[:10]
    rows, n = [], 0
    for a, b, run in windows(words, dur):
        xw = x[int(a * SR):int(b * SR)]
        rel = [{"s": w["s"] - a, "e": w["e"] - a} for w in run]
        for c in candidates(xw, rel, **rule):
            lab = label_pf(a + c["s"], a + c["e"], events, starts)
            if lab is None:
                continue
            dst = out / split / tag / f"{n:06d}.wav"
            write(dst, clip_of(xw, c))
            rows.append({"id": f"{corpus}/{tag}/{n:06d}", "corpus": corpus, "split": split, "label": lab,
                         "src": {"file": str(dst)}, "dur": round(c["e"] - c["s"], 3), "level_db": c["level_db"],
                         "episode": Path(wav).stem})
            n += 1
    return rows


def pf(args):
    root = args.podcastfillers
    csv_path = next(root.rglob("PodcastFillers.csv"))
    wav_root, tr_root = next(root.rglob("episode_wav")), next(root.rglob("episode_transcripts"))
    events = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            events[(r["episode_split_subset"], r["podcast_filename"])].append(
                (float(r["event_start_inepisode"]), float(r["event_end_inepisode"]), r["label_full_vocab"]))
    jobs = []
    for (split, ep), evs in sorted(events.items()):
        if split not in ("train", "validation"):
            continue                              # the test episodes are what scripts/eval_fillers.py scores
        wav, tr = wav_root / split / f"{ep}.wav", tr_root / split / f"{ep}.json"
        if wav.exists() and tr.exists():
            jobs.append((str(wav), str(tr), evs, split, args.clips / args.corpus, args.rule, args.corpus))
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for rs in ex.map(pf_episode, jobs):
            rows += rs
    return rows


def ami_words(x, words, split, out, corpus, meeting, every):
    """Filler words whole, and the tail of one other word in `every`, each cut out and padded as a candidate."""
    rows = []
    for i, w in enumerate(words):
        d = w["e"] - w["s"]
        if w["label"] == "filler":
            s = w["s"]
        elif int(hashlib.sha1(f"{meeting}/{i}".encode()).hexdigest(), 16) % every == 0 and d >= 0.16:
            s = w["e"] - max(0.08, d / 2)
        else:
            continue
        c = {"s": s, "e": min(w["e"], s + 2.5)}
        dst = out / split / meeting / f"w{i:06d}.wav"
        write(dst, clip_of(x, c))
        rows.append({"id": f"{corpus}/{meeting}/w{i:06d}", "corpus": corpus, "split": split, "label": w["label"],
                     "src": {"file": str(dst)}, "dur": round(c["e"] - c["s"], 3), "meeting": meeting,
                     "s": round(c["s"], 3), "e": round(c["e"], 3), "word": w["w"]})
    return rows


def ami_meeting(job):
    wav, words, sounds, split, out, rule, corpus, unmarked, only_unmarked, word_every = job
    from ami import union
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = resample(x.mean(1), sr)
    dur = len(x) / SR
    if word_every:
        return ami_words(x, words, split, out, corpus, Path(wav).name.split(".")[0], word_every)
    events = sorted([(w["s"], w["e"], w["label"]) for w in words] +
                    [(v["s"], v["e"], v["label"]) for v in sounds if v["e"] > v["s"]])
    starts = [ev[0] for ev in events]
    points = sorted((v["s"], v["label"]) for v in sounds if v["e"] <= v["s"])
    at = [t for t, _ in points]
    meeting = Path(wav).name.split(".")[0]
    rows, n = [], 0
    for a, b, run in windows(union(words), dur):
        xw = x[int(a * SR):int(b * SR)]
        rel = [{"s": w["s"] - a, "e": w["e"] - a} for w in run]
        cands = candidates(xw, rel, **rule)
        per_slot = Counter(c["slot"] for c in cands)
        for c in cands:
            lab = label_pf(a + c["s"], a + c["e"], events, starts, mapping=None)
            if lab is None and per_slot[c["slot"]] == 1:
                k = c["slot"]
                g0, g1 = (run[k - 1]["e"] if k else a), (run[k]["s"] if k < len(run) else b)
                marks = {points[q][1] for q in range(bisect.bisect_left(at, g0 - 0.05), bisect.bisect_right(at, g1 + 0.05))}
                lab = marks.pop() if len(marks) == 1 else None
            elif lab is None:
                k = c["slot"]
                g0, g1 = (run[k - 1]["e"] if k else a), (run[k]["s"] if k < len(run) else b)
                marks = range(bisect.bisect_left(at, g0 - 0.05), bisect.bisect_right(at, g1 + 0.05))
            if lab is not None and only_unmarked:
                continue
            if lab is None and unmarked and not len(marks) and not label_any(a + c["s"], a + c["e"], events, starts):
                lab = unmarked                    # no mark in its gap and nothing overlapping it
            if lab is None:
                continue
            dst = out / split / meeting / f"{n:06d}.wav"
            write(dst, clip_of(xw, c))
            rows.append({"id": f"{corpus}/{meeting}/{n:06d}", "corpus": corpus, "split": split, "label": lab,
                         "src": {"file": str(dst)}, "dur": round(c["e"] - c["s"], 3), "level_db": c["level_db"],
                         "meeting": meeting, "s": round(a + c["s"], 3), "e": round(a + c["e"], 3)})
            n += 1
    return rows


def ami(args):
    import zipfile
    from ami import ANNOTATIONS, meetings, read, split_of, wav_path
    jobs = []
    with zipfile.ZipFile(args.ami / ANNOTATIONS) as z:
        for m in meetings(z):
            wav = wav_path(args.ami, m)
            if not wav.exists():
                print(f"  [ami] no audio for {m}", flush=True)
                continue
            words, sounds = read(z, m)
            jobs.append((str(wav), words, sounds, split_of(m), args.clips / args.corpus, args.rule, args.corpus,
                         args.unmarked, args.only_unmarked, args.words))
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for rs in ex.map(ami_meeting, jobs):
            rows += rs
    return rows


def nvtts_rows(args):
    """Training clips outside the calibration draw, with the word times nonverbal/align_cache.py stored."""
    from gap_rule import load
    out = args.clips / args.corpus
    rows, stats = [], Counter()
    for cl, x in load(args.cache, "rest", args.calib_n):
        i, tags = cl["row"], [tuple(t) for t in cl["tags"]]
        split = "validation" if int(hashlib.sha1(str(i).encode()).hexdigest(), 16) % 10 == 0 else "train"
        for k, cand in enumerate(candidates(x, cl["words"], **args.rule)):
            exact = [c for c, sl in tags if sl == cand["slot"]]
            near = [c for c, sl in tags if abs(sl - cand["slot"]) <= 1]
            if exact:
                lab = NAME_OF.get(exact[0])
            elif near:
                lab = None                            # a word away from a tag: which sound it is, is unclear
            else:
                lab = "other"
            if lab is None or lab == "groan":
                stats["skipped"] += 1
                continue
            dst = out / split / f"{i:06d}_{k:02d}.wav"
            write(dst, clip_of(x, cand))
            rows.append({"id": f"{args.corpus}/{i:06d}_{k:02d}", "corpus": args.corpus, "split": split,
                         "label": lab, "src": {"file": str(dst)}, "dur": round(cand["e"] - cand["s"], 3),
                         "level_db": cand["level_db"]})
    print(f"  [nvtts] {len(rows)} candidates, {dict(stats)}", flush=True)
    return rows


def cap(rows, per_label, seed=0):
    """At most per_label rows of each (split, label): PodcastFillers' word tails would otherwise be most of it."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["label"])].append(r)
    out = []
    for k, rs in sorted(by.items()):
        rng.shuffle(rs)
        out += rs[:per_label]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["pf", "nvtts", "ami"])
    ap.add_argument("--podcastfillers", type=Path, default=Path(f"{MELATI_DATA}/bench/podcastfillers"))
    ap.add_argument("--ami", type=Path, default=Path(f"{MELATI_DATA}/nv/raw/ami"), help="nonverbal/ami.py's --root")
    ap.add_argument("--unmarked", default=None, help="ami: the label of a candidate nothing marks (default: left out)")
    ap.add_argument("--only-unmarked", action="store_true", help="ami: keep only those")
    ap.add_argument("--words", type=int, default=0, help="ami: word cuts instead of candidates, a speech tail from "
                                                          "one word in N")
    ap.add_argument("--cache", type=Path, default=Path(f"{MELATI_DATA}/nv/nvtts_align.json"),
                    help="NonverbalTTS word times (nonverbal/align_cache.py)")
    ap.add_argument("--rule", default="{}", help="melati.nv_gaps.candidates keyword arguments, as JSON")
    ap.add_argument("--corpus", default=None, help="the corpus name the rows get (default pf_cands / nvtts_cands / "
                                                   "ami_cands)")
    ap.add_argument("--calib-n", type=int, default=800)
    ap.add_argument("--clips", type=Path, default=Path("/dev/shm/melati_nv/clips"))
    ap.add_argument("--per-label", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.rule = json.loads(args.rule)
    args.corpus = args.corpus or f"{args.source}_cands"
    rows = {"pf": pf, "nvtts": nvtts_rows, "ami": ami}[args.source](args)
    found = Counter((r["split"], r["label"]) for r in rows)
    rows = cap(rows, args.per_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = Counter((r["split"], r["label"]) for r in rows)
    print(json.dumps({f"{s}/{l}": [found[(s, l)], kept[(s, l)]] for s, l in sorted(found)}, indent=1))
    print(f"CANDIDATES_DONE {args.source} {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
