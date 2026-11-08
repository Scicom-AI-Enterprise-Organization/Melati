"""Stage C: single-speaker segments, cut only in pauses. CPU.

Per recording, <work>/<rec>/segments.jsonl, one row per segment:
  {seg, rec, speaker, start, end, islands, forced_cut, overlap_s_removed, turn, prev, next}

Emilia-Pipe's segmentation (`cut_by_speaker_label` + Silero over pyannote turns) has three
defects this stage is built against (docs/emilia_bugs.md):
  1. a turn longer than 30 s is chopped every 30 s exactly, wherever that lands, so a word is
     cut in two whenever someone talks for more than half a minute;
  2. short pieces of the same speaker are glued across gaps of up to 2 s, gap included;
  3. regions where two speakers' turns overlap stay in both speakers' segments.
Here: Silero VAD finds speech islands in the speech stem; islands are given to the diarization
turn that owns them; anything inside an overlap of two turns (plus a 0.1 s margin) is removed;
islands are joined into a segment only across pauses of at most MAX_GAP, and a segment closes at
the first pause after it reaches PREFER_S or before it would pass MAX_S. A segment is only ever
cut inside an island when an island alone runs past MAX_S (no pause of MIN_PAUSE in 30 s of
speech), at the lowest-VAD-probability frame, and that is recorded as `forced_cut`.

    python -m melati.stage_seg --sources sources.jsonl --work $MELATI_DATA/podcast/work --workers 64
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR = 16000
MIN_PAUSE_MS = 100      # Silero min_silence: a pause this long separates two islands
MAX_GAP = 1.0           # never join islands across a longer pause
PREFER_S = 15.0         # close a segment at the first pause after this long...
MAX_S = 29.3            # ...and never let it pass this: with both pads a segment is at most 29.95 s, inside
                        # the 30 s window Whisper reads and the aligner accepts (it refuses anything longer)
MIN_S = 1.0             # shorter segments are dropped (export later flags < 3 s)
# context kept before the first and after the last island, never into the neighbour's speech. Silero
# (with its own 30 ms pad) ends an island before a soft final consonant or a breathy ending: with 0.10 s
# here, the pilot's segments ended inside their last word 17% of the time against Emilia-Pipe's 11%
# (scripts/compare_emilia.py, own transcript aligned to own audio), even where the next segment was a
# second away. Export trims to 0.15 s around the first and last aligned sound anyway.
PAD_LEAD_S = 0.25
PAD_TAIL_S = 0.40
OVERLAP_MARGIN = 0.10
# Where a removed overlap cuts an island, the speaker is mid-word: the pilot's segments ended inside their last
# word 30% of the time where an overlap had been removed, 10% elsewhere, and 64% of all such endings came from
# overlap removal. A side counts as cut only when the island ran on more than CUT_TOL_S into the overlap, and it
# moves back (at a start, forward) to the least speech-like frame within BACKOFF_S of it, at least EDGE_GUARD_S
# inside, with no padding past it. Turn edges are left alone: an island spanning two turns is two speakers taking
# turns without a pause, and backing off there too cost 27% of the pilot's speech. Requiring a true pause at the
# new edge (rather than the lowest frame nearby) cost 57%: Silero rarely dips inside continuous speech.
CUT_TOL_S = 0.15
BACKOFF_S = 1.5
EDGE_GUARD_S = 0.15


def vad(x: np.ndarray):
    """Speech islands [(start, end)] in seconds and the frame probabilities (32 ms frames)."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    model = load_silero_vad()
    wav = torch.from_numpy(x)
    ts = get_speech_timestamps(wav, model, sampling_rate=SR, threshold=0.5, min_speech_duration_ms=150,
                               min_silence_duration_ms=MIN_PAUSE_MS, speech_pad_ms=30, return_seconds=False)
    model.reset_states()
    probs = model.audio_forward(wav, sr=SR).squeeze(0).numpy()    # one value per 512 samples
    return [(t["start"] / SR, t["end"] / SR) for t in ts], probs


def subtract(ivs, cuts, islands=None):
    """Intervals minus cut spans. Each interval is (s, e) or (s, e, cut_start, cut_end); the result is always the
    flagged form, a side flagged True when a cut made it while speech ran on more than CUT_TOL_S past it (the
    speech is the interval itself, or with `islands` the VAD island holding it)."""
    def run_on(t, forward):
        """How far speech continues past time t, in the island containing it (or the interval)."""
        for a, b in islands or []:
            if a <= t <= b:
                return (b - t) if forward else (t - a)
        return None

    out = []
    for iv in ivs:
        s, e = iv[0], iv[1]
        cs, ce = (iv[2], iv[3]) if len(iv) == 4 else (False, False)
        pieces = [(s, e, cs, ce)]
        for a, b in cuts:
            nxt = []
            for p, q, ps, pe in pieces:
                if b <= p or a >= q:
                    nxt.append((p, q, ps, pe))
                    continue
                if a > p:
                    r = run_on(a, True)
                    nxt.append((p, a, ps, (q - a if r is None else r) > CUT_TOL_S))
                if b < q:
                    r = run_on(b, False)
                    nxt.append((b, q, (b - p if r is None else r) > CUT_TOL_S, pe))
            pieces = nxt
        out += pieces
    return out


def back_off(piece, probs):
    """A piece whose start or end was made by a cut, each cut side moved to the least speech-like frame within
    BACKOFF_S of it (at least EDGE_GUARD_S inside); None when too little is left. -> (s, e, cut_start, cut_end),
    the flags kept so padding knows not to reach past a moved side."""
    s, e, cs, ce = piece
    if not (cs or ce):
        return piece
    hop = 512 / SR
    g, w = max(1, int(round(EDGE_GUARD_S / hop))), int(round(BACKOFF_S / hop))
    if ce:
        f1 = min(len(probs), int(round(e / hop)))
        lo, hi = max(int(round(s / hop)) + g, f1 - w), f1 - g
        if hi <= lo:
            return None
        e = (lo + int(np.argmin(probs[lo:hi])) + 1) * hop
    if cs:
        f0 = int(round(s / hop))
        lo, hi = f0 + g, min(int(round(e / hop)) - g, f0 + w)
        if hi <= lo:
            return None
        s = (lo + int(np.argmin(probs[lo:hi]))) * hop
    return (s, e, cs, ce) if e - s > 0.3 else None


def overlaps(turns):
    """Spans where two different speakers' turns overlap."""
    ev = sorted([(s, 1, k) for s, e, k in turns] + [(e, -1, k) for s, e, k in turns])
    active, out, start = {}, [], None
    for t, d, k in ev:
        before = len([v for v in active.values() if v > 0])
        active[k] = active.get(k, 0) + d
        after = len([v for v in active.values() if v > 0])
        if before < 2 <= after:
            start = t
        elif before >= 2 > after and start is not None:
            if t > start:
                out.append((start, t))
            start = None
    return out


def split_long(island, probs, max_s=MAX_S):
    """An island longer than max_s is cut at its lowest-probability frame, recursively."""
    s, e = island
    if e - s <= max_s:
        return [(s, e, False)]
    lo, hi = int((s + 0.5 * max_s / 2) * SR / 512), int((s + max_s) * SR / 512)
    hi = min(hi, len(probs) - 1)
    if hi <= lo:
        cut = s + max_s
    else:
        cut = (lo + int(np.argmin(probs[lo:hi]))) * 512 / SR
    head = [(s, cut, True)]
    return head + [(a, b, True) for a, b, _ in split_long((cut, e), probs, max_s)]


def merge_same_speaker(turns):
    """One speaker's overlapping or nested turns become one turn. The diarizer sometimes emits a turn inside
    another of the same speaker ([1151.00, 1154.69] and [1152.41, 1153.88], both S01); `overlaps` only sees
    two DIFFERENT speakers, so both became segments, and the padding below then cut the outer one short at
    the inner one's start and gave the inner one a negative length."""
    by = {}
    for s, e, k in sorted(turns):
        cur = by.setdefault(k, [])
        if cur and s < cur[-1][1]:
            cur[-1][1] = max(cur[-1][1], e)
        else:
            cur.append([s, e])
    return sorted((s, e, k) for k, ivs in by.items() for s, e in ivs)


def segment(rec_id: str, x: np.ndarray, turns, islands=None, probs=None):
    dur = len(x) / SR
    n_turns = len(turns)
    turns = merge_same_speaker(turns)
    if islands is None:
        islands, probs = vad(x)
    ovl = [(max(0.0, a - OVERLAP_MARGIN), min(dur, b + OVERLAP_MARGIN)) for a, b in overlaps(turns)]
    segs, backed_off, lost_s = [], 0, 0.0
    for ti, (ts, te, spk) in enumerate(sorted(turns)):
        own = [(max(s, ts), min(e, te)) for s, e in islands if min(e, te) - max(s, ts) > 0.05]
        clean = []
        for piece in subtract(own, ovl, islands=islands):
            b = back_off(piece, probs)
            if b is not piece:
                backed_off += 1
                lost_s += (piece[1] - piece[0]) - ((b[1] - b[0]) if b else 0.0)
            if b:
                clean.append(b)
        removed = sum(e - s for s, e, *_ in own) - sum(e - s for s, e, *_ in clean)
        pieces = []
        for s, e, cs, ce in clean:
            parts = split_long((s, e), probs)
            # the cut flags belong to the piece's outer edges, not to the forced cuts split_long adds inside
            pieces += [(a, b, f, cs and k == 0, ce and k == len(parts) - 1) for k, (a, b, f) in enumerate(parts)]
        cur = []
        for s, e, forced, cs, ce in pieces:
            if cur:
                length, gap = e - cur[0][0], s - cur[-1][1]
                # never join across a removed overlap or a turn edge: the gap holds someone else's speech
                if (gap > MAX_GAP or length > MAX_S or cs or cur[-1][4]
                        or (cur[-1][1] - cur[0][0] >= PREFER_S and gap >= 0.3)):
                    segs.append((spk, ti, cur, removed))
                    removed = 0.0
                    cur = []
            cur.append((s, e, forced, cs, ce))
        if cur:
            segs.append((spk, ti, cur, removed))
    out = []
    for spk, ti, isl, removed in segs:
        s, e = isl[0][0], isl[-1][1]
        if e - s < MIN_S:
            continue
        out.append({"rec": rec_id, "speaker": spk, "turn": ti,
                    "islands": [[round(a, 3), round(b, 3)] for a, b, *_ in isl],
                    "speech_start": round(s, 3), "speech_end": round(e, 3),
                    "start_backed_off": bool(isl[0][3]), "end_backed_off": bool(isl[-1][4]),
                    "forced_cut": any(f for _, _, f, *_ in isl), "overlap_s_removed": round(removed, 3)})
    out.sort(key=lambda r: r["speech_start"])
    # no two segments may share speech: padding below clamps a segment to its neighbour's speech, so an
    # overlap would cut speech off. After the merge above only a diarizer anomaly can cause one; drop it.
    kept, dropped = [], 0
    for r in out:
        if kept and r["speech_start"] < kept[-1]["speech_end"]:
            dropped += 1
            continue
        kept.append(r)
    out = kept
    # padding never reaches into the neighbouring segment or a removed overlap, and none is added past a side that
    # was moved off a cut: what lies beyond it is the rest of the word
    for k, r in enumerate(out):
        lo = max([out[k - 1]["speech_end"] if k else 0.0] + [b for a, b in ovl if b <= r["speech_start"]])
        hi = min([out[k + 1]["speech_start"] if k + 1 < len(out) else dur] + [a for a, b in ovl if a >= r["speech_end"]])
        lead = 0.0 if r["start_backed_off"] else PAD_LEAD_S
        tail = 0.0 if r["end_backed_off"] else PAD_TAIL_S
        r["start"] = round(max(lo, r["speech_start"] - lead, 0.0), 3)
        r["end"] = round(min(hi, r["speech_end"] + tail, dur), 3)
        r["seg"] = f"{rec_id}_{k:04d}"
        r["prev"] = out[k - 1]["seg"] if k else None
    for k, r in enumerate(out):
        r["next"] = out[k + 1]["seg"] if k + 1 < len(out) else None
    return out, {"vad_speech_s": round(sum(b - a for a, b in islands), 2),
                 "overlap_s": round(sum(b - a for a, b in ovl), 2), "segments": len(out),
                 "forced_cuts": sum(r["forced_cut"] for r in out),
                 "pieces_backed_off": backed_off, "speech_s_lost_to_back_off": round(lost_s, 2),
                 "turns_merged": n_turns - len(turns), "overlapping_dropped": dropped}


def run_one(args):
    work, rec_id = args
    import soundfile as sf
    d = Path(work) / rec_id
    if (d / "segments.jsonl").exists():
        return rec_id, None
    try:
        x, sr = sf.read(str(d / "speech16k.flac"), dtype="float32")
        assert sr == SR
        turns = [(s, e, k) for s, e, k in json.loads((d / "diar.json").read_text())["turns"]]
        segs, stats = segment(rec_id, x, turns)
    except Exception as e:  # noqa: BLE001
        return rec_id, f"{type(e).__name__}: {str(e)[:160]}"
    tmp = d / "segments.jsonl.part"
    with tmp.open("w") as fh:
        for r in segs:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(d / "segments.jsonl")
    (d / "seg_stats.json").write_text(json.dumps(stats))
    return rec_id, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    recs = [json.loads(l)["id"] for l in args.sources.open()]
    recs = [r for r in recs if (args.work / r / "diar.json").exists() and not (args.work / r / "segments.jsonl").exists()]
    if args.limit:
        recs = recs[: args.limit]
    print(f"[seg] {len(recs)} recordings", flush=True)
    errors = 0
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for k, (rid, err) in enumerate(ex.map(run_one, [(str(args.work), r) for r in recs], chunksize=4)):
            if err:
                errors += 1
                print(f"[seg] {rid}: {err}", flush=True)
            if (k + 1) % 500 == 0:
                print(f"[seg] {k + 1}/{len(recs)} errors={errors}", flush=True)
    print(f"[seg] done, errors={errors}", flush=True)


def _init():
    import torch
    torch.set_num_threads(1)


if __name__ == "__main__":
    main()
