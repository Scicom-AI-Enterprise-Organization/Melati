"""The Melati stages as coroutines: CPU work in local pools, every model call through melati.client.

Nothing in this module touches a GPU. Each stage reads the previous stages' files in
<work>/<rec>/, calls the model servers, and writes its own file atomically; a stage whose file
exists is skipped, so a run resumes wherever it stopped. melati/run.py drives many recordings at
once through all stages, which is what keeps every server's batches full: while one recording is
being separated, others are being transcribed, aligned and scored.

  front    decode, one gain per recording, PANNs events, DBS separation only where background;
           the separated version every later stage reads is speech + effects ("vocal"): DBS-TIGER
           files laughs, coughs and breaths under EFFECTS (scripts/nv_injection.py: 98-100% of a
           vocalisation's energy), so its speech stem alone deletes what Melati exists to tag
  diar     who spoke when (vLLM, Who-Spoke-When-Over-the-Phone)
  seg      single-speaker segments cut only in pauses, overlap removed (Silero, CPU)
  asr      transcripts (vLLM Whisper), re-decoded when the detected language is not allowed
  align    word times with confidence (Where-Words-Begin-and-End)
  quality  SOTA-MOS and DNSMOS, on the separated speech and on the mix
  tag      emotion (phrases and segments), voiced gaps, rate spans, and non-verbal sounds in the word
           gaps labelled by Melati's non-verbal heads (melati.nv_gaps), or by Qwen3-Omni when the cluster
           serves `omni` and no head; and each segment's laughter and cough probability every 0.1 s
           (melati.nv_frames), which reaches the sounds over speech that no gap holds
  sounds   which of AudioSet's 527 sounds each segment holds, from Melati's own AudioSet tagger
           (Scicom-intl/Melati-AudioSet-EAT-Large, serve/sounds_server.py), on the same speech and effects
  fill     each voiced gap transcribed alone: filler, dropped word, or nothing
  export   FLAC + one manifest row per segment (melati.stage_export.build)
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from melati import audio as A
from melati import diarize as DZ
from melati import emotion as EMO
from melati import gaps as G
from melati import nonverbal as NV
from melati import nv_gaps as NG
from melati import sed as SED

SR16, SR24, SR44 = 16000, 24000, 44100
STAGES = ("front", "diar", "seg", "asr", "align", "quality", "tag", "sounds", "fill", "export")
OUTPUT = {"front": "sep.json", "diar": "diar.json", "seg": "segments.jsonl", "asr": "asr.jsonl",
          "align": "align.jsonl", "quality": "mos.jsonl", "tag": "tags.jsonl", "sounds": "sounds.jsonl",
          "fill": "fill.jsonl", "export": "export.jsonl"}
# the services each stage calls through melati.client; seg and export need none
SERVICES = {"front": ("front",), "diar": ("diar",), "seg": (), "asr": ("asr",), "align": ("align",),
            "quality": ("mos", "tag"), "tag": ("tag",), "sounds": ("sounds",), "fill": ("asr",), "export": ()}

# Stage A separates only where PANNs hears background (music, singing, crowd, broadcast media, noise):
# DBS-TIGER runs at 14 to 15x realtime per GPU whatever the batch (scripts/profile_sep.py), about 160
# GPU-hours for the podcast corpus if applied everywhere, and where there is nothing to remove the
# recording itself is cleaner than any separator's output (scripts/nv_injection.py: what it does to a cough).
BG_THR, BG_DILATE_S, CONTEXT_S, FADE_S = 0.30, 1.5, 3.0, 0.05


@dataclass
class Ctx:
    work: Path
    client: object
    cpu: object                      # ProcessPoolExecutor
    io: object                       # ThreadPoolExecutor
    allowed: tuple = ("ms", "en", "zh", "ta")
    emotion_models: tuple = ('h_whisper', 'h_whisper_ft', 'h_w2vbert', 'h_wavlm', 'h_e2v', 'meralion', 'e2v')
    nv_models: tuple = ('nv_whisper',)   # the non-verbal head (nonverbal/); Qwen3-Omni if the cluster serves none
    nv_frame_models: tuple = ('nvf_whisper',)   # the frame-level detector (melati.nv_frames); none if not served
    out: Path | None = None          # export root
    export_cfg: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    async def cpu_(self, fn, *a):
        return await asyncio.get_running_loop().run_in_executor(self.cpu, fn, *a)

    async def io_(self, fn, *a):
        return await asyncio.get_running_loop().run_in_executor(self.io, fn, *a)


def write_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    tmp.replace(path)


def write_jsonl(path: Path, rows):
    write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def by_seg(path: Path):
    return {r["seg"]: r for r in read_jsonl(path)}


# ----------------------------------------------------------------------------- A. front
def _front_load(rec):
    """Decode, resample to the separator's rate, and one gain for the whole recording."""
    x, sr = A.decode(Path(rec["path"]))
    x = A.resample(x, sr, SR44)
    lufs_in = A.loudness(x, SR44)
    gain = A.recording_gain_db(x, SR44)
    return rec, A.apply_gain(x, gain), {"native_sr": sr, "duration_s": round(len(x) / SR44, 3),
                                         "lufs_in": round(lufs_in, 2) if np.isfinite(lufs_in) else None,
                                         "gain_db": round(gain, 2)}


def splice(base: np.ndarray, part: np.ndarray, a: int, b: int, fade: int) -> None:
    """base[a:b] <- part, with linear cross-fades of `fade` samples at both edges (in place)."""
    seg_ = base[a:b].copy()
    w = np.ones(b - a, np.float32)
    f = min(fade, (b - a) // 2)
    if f > 0:
        if a > 0:
            w[:f] = np.linspace(0, 1, f, dtype=np.float32)
        if b < len(base):
            w[-f:] = np.linspace(1, 0, f, dtype=np.float32)
    base[a:b] = w * part + (1 - w) * seg_


def _front_save(d: Path, rec, mix, tracks, speech, vocal, music, sfx, meta):
    d.mkdir(parents=True, exist_ok=True)
    A.write_flac(d / "speech24k.flac", A.resample(speech, SR44, SR24), SR24)
    A.write_flac(d / "speech16k.flac", A.resample(speech, SR44, SR16), SR16)
    A.write_flac(d / "vocal16k.flac", A.resample(vocal, SR44, SR16), SR16)
    A.write_flac(d / "vocal24k.flac", A.resample(vocal, SR44, SR24), SR24)
    np.savez_compressed(d / "sed.npz", fps=SED.FPS, **{g: v.astype(np.float16) for g, v in tracks.items()})
    et = lambda x: A.frame_energy_db(x, SR44, 0.05).astype(np.float16)
    np.savez_compressed(d / "stems.npz", frame_s=0.05, mix=et(mix), music=et(music), sfx=et(sfx), speech=et(speech))
    meta = dict(meta, source=rec["path"], id=rec["id"], sep_model="JusperLee/DBS-TIGER-DnR",
                sed_model="PANNs Cnn14_DecisionLevelMax", lufs_out=round(A.loudness(mix, SR44), 2))
    write_atomic(d / "sep.json", json.dumps(meta))


async def front(ctx: Ctx, rec):
    d = ctx.work / rec["id"]
    _, mix, meta = await ctx.cpu_(_front_load, rec)
    x32 = await ctx.cpu_(A.resample, mix, SR44, SED.SR)
    sed = await ctx.client.sed(x32)
    fps = sed.pop("fps")
    bg = np.max(np.stack([sed[g] for g in SED.BACKGROUND if g in sed]), 0)
    regs = SED.regions(bg, BG_THR, dilate_s=BG_DILATE_S, fps=fps)
    speech, vocal = mix.copy(), mix.copy()
    music, sfx = np.zeros_like(mix), np.zeros_like(mix)

    async def one(ra, rb):
        a, b = int(ra * SR44), min(len(mix), int(rb * SR44))
        ca, cb = max(0, a - int(CONTEXT_S * SR44)), min(len(mix), b + int(CONTEXT_S * SR44))
        return a, b, ca, await ctx.client.separate(mix[ca:cb])

    for a, b, ca, st in await asyncio.gather(*[one(ra, rb) for ra, rb in regs]):
        lo, hi = a - ca, b - ca
        splice(speech, st[2][lo:hi], a, b, int(FADE_S * SR44))
        splice(vocal, st[1][lo:hi] + st[2][lo:hi], a, b, int(FADE_S * SR44))
        music[a:b], sfx[a:b] = st[0][lo:hi], st[1][lo:hi]
    sep_s = sum(b - a for a, b in regs)
    meta = dict(meta, regions=[[round(a, 2), round(b, 2)] for a, b in regs], separated_s=round(sep_s, 2),
                separated_frac=round(sep_s / max(meta["duration_s"], 1e-9), 4))
    await ctx.io_(_front_save, d, rec, mix, sed, speech, vocal, music, sfx, meta)


# ----------------------------------------------------------------------------- B. diar
async def diar(ctx: Ctx, rec):
    d = ctx.work / rec["id"]
    path = d / "speech16k.flac"
    dur = sf.info(str(path)).frames / SR16
    turns, mode, hit, err = None, "one", False, None
    if dur <= DZ.ONE_PASS_MAX_S:
        try:
            text, hit = await ctx.client.diarize(path, dur, tries=2)
        except RuntimeError as ex:           # no answer in two half-hour tries: windows instead
            text, err = None, str(ex)[:300]
        if text is not None and not DZ.failed(text, hit, dur):
            turns = [[round(s, 3), round(e, 3), f"S{g:02d}"] for s, e, g in sorted(DZ.turns_of(text))]
    if turns is None:
        mode = "windows"
        wins = DZ.plan(dur)
        x, _ = sf.read(str(path), dtype="float32")
        wdir = d / "diar_windows"
        wdir.mkdir(exist_ok=True)
        paths = []
        for k, (s, e) in enumerate(wins):
            p = wdir / f"w{k:03d}.flac"
            await ctx.io_(A.write_flac, p, x[int(s * SR16):int(e * SR16)], SR16)
            paths.append(p)
        texts = [t for t, _ in await asyncio.gather(*[ctx.client.diarize(p, e - s) for p, (s, e) in zip(paths, wins)])]
        turns = DZ.stitch(wins, texts)
        for p in paths:
            p.unlink(missing_ok=True)
    write_atomic(d / "diar.json", json.dumps({"turns": turns, "mode": mode, "budget_hit": hit,
                                              **({"one_pass_error": err} if err else {}),
                                              "model": "Who-Spoke-When-Over-the-Phone-Whisper-Large-V3-Merged",
                                              "duration_s": round(dur, 3)}))


# ----------------------------------------------------------------------------- C. seg
def _seg(work: str, rid: str):
    import torch
    torch.set_num_threads(1)
    from melati.stage_seg import run_one
    return run_one((work, rid))


async def seg(ctx: Ctx, rec):
    rid, err = await ctx.cpu_(_seg, str(ctx.work), rec["id"])
    if err:
        raise RuntimeError(err)


# ----------------------------------------------------------------------------- D. asr
# A loop guard, Whisper's own: the transcript is re-decoded at rising temperature when it compresses like a
# loop (gzip ratio over 2.4, openai/whisper's compression_ratio_threshold), keeping the first that does not,
# and every decode is capped at a budget proportional to the clip, so a loop that survives is short. The
# fine-tunes loop on repeated syllables ("ta ta ta" written until the 440-token limit), and podcast laughter
# is repeated syllables.
TOKENS_PER_S, TOKENS_MIN = 15.0, 32     # fast speech is 6-10 tokens/s, so real speech never meets the cap
LOOP_GZIP = 2.4
FALLBACK_T = (0.2, 0.4, 0.6)


def gzip_ratio(text: str) -> float:
    import zlib
    b = (text or "").encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


async def decode(client, x, language=None) -> dict:
    """Whisper on vLLM, greedy, with the loop guard above; -> the response plus `fallback_t` (0 when the
    greedy transcript was kept) and `gzip_ratio` of the one kept."""
    cap = int(TOKENS_MIN + TOKENS_PER_S * len(x) / SR16)
    resp = await client.transcribe(x, language, max_tokens=cap)
    used = 0.0
    for t in FALLBACK_T:
        if gzip_ratio(resp.get("text") or "") <= LOOP_GZIP:
            break
        resp = await client.transcribe(x, language, temperature=t, max_tokens=cap, seed=0)
        used = t
    return dict(resp, fallback_t=used, gzip_ratio=round(gzip_ratio(resp.get("text") or ""), 3))


def _summary(resp: dict) -> dict:
    segs = resp.get("segments") or []
    lp = [s.get("avg_logprob") for s in segs if s.get("avg_logprob") is not None]
    cr = [s.get("compression_ratio") for s in segs if s.get("compression_ratio") is not None]
    return {"text": (resp.get("text") or "").strip(), "language": resp.get("language"),
            "avg_logprob": round(float(np.mean(lp)), 4) if lp else None,
            "compression_ratio": round(float(np.max(cr)), 3) if cr else None,
            "fallback_t": resp.get("fallback_t"), "gzip_ratio": resp.get("gzip_ratio"),
            "looped": (resp.get("gzip_ratio") or 0) > LOOP_GZIP}


async def transcribe(ctx: Ctx, x):
    """One segment: the model picks the language; outside the allowed set, every allowed language
    is tried and the most likely transcript kept (the first detection is kept too)."""
    first = _summary(await decode(ctx.client, x))
    out = dict(first, detected_language=first["language"])
    if ctx.allowed and first["language"] not in ctx.allowed:
        cands = await asyncio.gather(*[decode(ctx.client, x, lang) for lang in ctx.allowed])
        best = max(((lang, _summary(r)) for lang, r in zip(ctx.allowed, cands)),
                   key=lambda lr: lr[1]["avg_logprob"] if lr[1]["avg_logprob"] is not None else -99)
        out = dict(best[1], language=best[0], detected_language=first["language"])
    return out


async def asr(ctx: Ctx, rec):
    d = ctx.work / rec["id"]
    x, _ = sf.read(str(d / "speech16k.flac"), dtype="float32")
    segs = read_jsonl(d / "segments.jsonl")
    res = await asyncio.gather(*[transcribe(ctx, x[int(s["start"] * SR16):int(s["end"] * SR16)]) for s in segs])
    write_jsonl(d / "asr.jsonl", [dict(r, seg=s["seg"]) for s, r in zip(segs, res)])


# ----------------------------------------------------------------------------- E. align
WHISPER_NAMES = {"english": "en", "malay": "ms", "chinese": "zh", "indonesian": "id", "tamil": "ta",
                 "cantonese": "yue", "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi"}


def lang_code(x):
    """vLLM may report a language by name; the aligner wants Whisper's code."""
    if not x:
        return None
    x = x.lower()
    return WHISPER_NAMES.get(x, x if len(x) <= 3 else None)


async def align(ctx: Ctx, rec, per_request: int = 16):
    d = ctx.work / rec["id"]
    x, _ = sf.read(str(d / "speech16k.flac"), dtype="float32")
    segs = {s["seg"]: s for s in read_jsonl(d / "segments.jsonl")}
    items = [(a["seg"], x[int(segs[a["seg"]]["start"] * SR16):int(segs[a["seg"]]["end"] * SR16)], a["text"],
              lang_code(a.get("language"))) for a in read_jsonl(d / "asr.jsonl") if a.get("text")]
    chunks = [items[i:i + per_request] for i in range(0, len(items), per_request)]
    results = await asyncio.gather(*[ctx.client.align([(c[1], c[2], c[3]) for c in ch]) for ch in chunks])
    rows = []
    for ch, res in zip(chunks, results):
        for (sid, xx, _, _), r in zip(ch, res):
            words = r.get("words") or []
            rows.append({"seg": sid, "words": words, "score": r.get("score"), "error": r.get("error"),
                         "first_word_s": words[0]["s"] if words else None,
                         "tail_s": round(len(xx) / SR16 - words[-1]["e"], 3) if words else None})
    write_jsonl(d / "align.jsonl", rows)


# ----------------------------------------------------------------------------- F. quality
def _mix24(rec, gain_db, n):
    mix, _ = A.decode(Path(rec["path"]), sr=SR24)
    return A.apply_gain(mix, gain_db)[:n]


async def quality(ctx: Ctx, rec, per_request: int = 16):
    """SOTA-MOS and DNSMOS of every segment twice: the separated version (speech + effects, which is
    what export would write) and the original mix."""
    d = ctx.work / rec["id"]
    meta = json.loads((d / "sep.json").read_text())
    vocal, _ = sf.read(str(d / "vocal24k.flac"), dtype="float32")
    mix = await ctx.cpu_(_mix24, rec, meta["gain_db"], len(vocal))
    segs = read_jsonl(d / "segments.jsonl")
    cut = lambda x, s: x[int(s["start"] * SR24):int(s["end"] * SR24)]
    sp, mx = [cut(vocal, s) for s in segs], [cut(mix, s) for s in segs]
    mos_sp, mos_mx = await asyncio.gather(asyncio.gather(*[ctx.client.mos(x) for x in sp]),
                                          asyncio.gather(*[ctx.client.mos(x) for x in mx]))
    to16 = lambda xs: [A.resample(x, SR24, SR16) for x in xs]
    async def dns(xs):
        xs16 = await ctx.io_(to16, xs)
        out = []
        for i in range(0, len(xs16), per_request):
            out += await ctx.client.dnsmos(xs16[i:i + per_request])
        return out
    dns_sp, dns_mx = await asyncio.gather(dns(sp), dns(mx))
    write_jsonl(d / "mos.jsonl", [{"seg": s["seg"], "mos_vocal": round(a, 4), "mos_mix": round(b, 4),
                                   "dnsmos_vocal": c, "dnsmos_mix": e}
                                  for s, a, b, c, e in zip(segs, mos_sp, mos_mx, dns_sp, dns_mx)])


# ----------------------------------------------------------------------------- G. tag
async def tag(ctx: Ctx, rec, per_request: int = 32):
    d = ctx.work / rec["id"]
    speech, _ = sf.read(str(d / "vocal16k.flac"), dtype="float32")      # speech + effects: keeps the laughs
    tracks = dict(np.load(d / "sed.npz"))
    fps = int(tracks.pop("fps"))
    segs = read_jsonl(d / "segments.jsonl")
    al = by_seg(d / "align.jsonl")
    out = {s["seg"]: {"seg": s["seg"]} for s in segs}
    clips, owner = [], []
    nv_jobs = []                                   # (segment id, its audio, its gap candidates)
    omni = ctx.client.ep.count("omni") > 0
    for s in segs:
        x = speech[int(s["start"] * SR16):int(s["end"] * SR16)]
        words = (al.get(s["seg"]) or {}).get("words") or []
        o = out[s["seg"]]
        o["sed"] = NV.events_from_tracks(tracks, fps, s["start"], s["end"])
        o["gaps"] = G.voiced_gaps(words, x, SR16)
        o["rate"] = [list(r) for r in G.rate_spans(words)]
        if (ctx.nv_models or omni) and words:
            cands = NG.candidates(x, words)
            if cands:
                nv_jobs.append((s["seg"], x, cands))
        if len(x) >= int(0.3 * SR16):
            clips.append(x); owner.append((s["seg"], None))
        for a, b, ps, pe in EMO.phrases(words):
            c = x[int(ps * SR16):int(pe * SR16)]
            if len(c) >= int(0.3 * SR16):
                clips.append(c); owner.append((s["seg"], (a, b, ps, pe)))
    order = sorted(range(len(clips)), key=lambda j: len(clips[j]))
    for model in ctx.emotion_models:
        chunks = [order[i:i + per_request] for i in range(0, len(order), per_request)]
        try:
            res = await asyncio.gather(*[ctx.client.emotion(model, [clips[j] for j in ch]) for ch in chunks])
        except ValueError as e:                           # a model this cluster does not serve
            print(f"[tag] {rec['id']}: emotion/{model} skipped: {str(e)[:120]}", flush=True)
            continue
        for ch, probs in zip(chunks, res):
            for j, p in zip(ch, probs):
                sid, ph = owner[j]
                e = out[sid].setdefault("emotion", {}).setdefault(model, {"segment": None, "phrases": []})
                pr = [round(float(v), 4) for v in p]
                if ph is None:
                    e["segment"] = pr
                else:
                    e["phrases"].append([ph[0], ph[1], round(ph[2], 3), round(ph[3], 3), pr])
    if nv_jobs:                                    # each candidate asked about on its own (melati.nv_gaps)
        labelled = None
        if ctx.nv_models:                          # Melati's own heads, served beside the emotion ones
            try:
                labelled = await asyncio.gather(*[NG.label_heads(ctx.client, x, c, ctx.nv_models)
                                                  for _, x, c in nv_jobs])
            except ValueError as e:                # a cluster that serves no non-verbal heads
                print(f"[tag] {rec['id']}: non-verbal heads skipped: {str(e)[:120]}", flush=True)
        if labelled is None and omni:
            labelled = await asyncio.gather(*[NG.label(ctx.client, x, c) for _, x, c in nv_jobs])
        for (sid, _, _), lab in zip(nv_jobs, labelled or []):
            out[sid]["nv_gaps"] = lab
    await tag_frames(ctx, rec, speech, segs, out, per_request)
    write_jsonl(d / "tags.jsonl", [out[s["seg"]] for s in segs])


FRAME_CLASSES = ("laugh", "cough")          # the frame classes that can become tags (melati.stage_export)


async def tag_frames(ctx: Ctx, rec, speech, segs, out, per_request: int = 32):
    """Each segment's laughter and cough probabilities every 1/fps s, from the frame-level detector, into
    out[seg]["nv_frames"] = {model, fps, laugh: [...], cough: [...]}."""
    clips = [(s["seg"], speech[int(s["start"] * SR16):int(s["end"] * SR16)]) for s in segs]
    clips = [(sid, x) for sid, x in clips if len(x) >= int(0.3 * SR16)]
    for model in ctx.nv_frame_models:
        chunks = [clips[i:i + per_request] for i in range(0, len(clips), per_request)]
        try:
            res = await asyncio.gather(*[ctx.client.frames(model, [x for _, x in ch]) for ch in chunks])
        except ValueError as e:                    # a cluster that serves no frame model
            print(f"[tag] {rec['id']}: frames/{model} skipped: {str(e)[:120]}", flush=True)
            continue
        for ch, (arrs, classes, fps) in zip(chunks, res):
            for (sid, _), a in zip(ch, arrs):
                out[sid]["nv_frames"] = {"model": model, "fps": fps,
                                         **{c: [round(float(v), 3) for v in a[:, classes.index(c)]]
                                            for c in FRAME_CLASSES}}
        break                                      # one frame model is enough


# ----------------------------------------------------------------------------- sounds
# The tagger reads one window: 10.24 s for an EAT model (1024 frames of 10 ms), 10 s for the ensemble; the server
# says which (/health). A longer segment is read in such windows every half
# window, the last one ending with the segment, and a class keeps its highest probability over them: a sound anywhere
# in the segment. Probabilities under SOUNDS_FLOOR are not kept; the export's threshold is a config value above it.
# A class of the inline vocabulary (melati/sounds_inline.json, audioset/sounds_vocab.py) that the tagger hears at
# SOUNDS_FRAME_GATE or more is also placed in time, every 160 ms, by the two frame models the server averages; the
# export's own gates (stage_export `sounds_inline_gates`: 0.5, and 0.8 for Music) sit above this one, so they can be
# raised without a re-run.
SOUNDS_FLOOR, SOUNDS_FRAME_GATE = 0.01, 0.3
SOUNDS_INLINE = json.loads((Path(__file__).with_name("sounds_inline.json")).read_text())


def _read16(path: Path):
    return sf.read(str(path), dtype="float32")


def windows(n: int, win: int, hop: int) -> list[int]:
    """The start samples of the windows a clip of n samples is read in: one if it fits, else every hop, and a last
    one that ends with the clip."""
    return [0] if n <= win else list(range(0, n - win, hop)) + [n - win]


async def sounds(ctx: Ctx, rec, per_request: int = 32):
    """Each segment's AudioSet sounds into sounds.jsonl: {seg, model, windows, p: {label: probability}}, the labels
    in falling probability, and `frames` {model, step_s, label: [probability every step]} for the inline vocabulary's
    classes it holds at SOUNDS_FRAME_GATE or more. It reads vocal16k.flac, the speech-and-effects version every tag
    reads and most rows ship (music removed where the front stage separated). A server without the frame models
    (serve/sounds_server.py, FRAMES_CKPT) gives no `frames`, and the export then writes no inline sound tags."""
    d = ctx.work / rec["id"]
    # in the io pool: read in the event loop, an hour of FLAC off NFS held every other recording in flight
    # (a 7,600-recording backfill ran at one replica's pace with five idle)
    x, _ = await ctx.io_(_read16, d / "vocal16k.flac")
    segs = read_jsonl(d / "segments.jsonl")
    win_s = await ctx.client.sounds_window()
    win, hop = int(win_s * SR16), int(win_s / 2 * SR16)
    clips, owner = [], []
    for k, s in enumerate(segs):
        a, b = int(s["start"] * SR16), int(s["end"] * SR16)
        for w0 in windows(b - a, win, hop):
            clips.append(x[a + w0:a + min(b - a, w0 + win)]); owner.append(k)
    chunks = [range(i, min(i + per_request, len(clips))) for i in range(0, len(clips), per_request)]
    res = await asyncio.gather(*[ctx.client.sounds([clips[j] for j in ch]) for ch in chunks])
    labels, model = (res[0][1], res[0][2]) if res else ([], None)
    best = np.zeros((len(segs), len(labels)), np.float32)
    for ch, (probs, _, _) in zip(chunks, res):
        for j, p in zip(ch, probs):
            np.maximum(best[owner[j]], p, out=best[owner[j]])
    n = np.bincount(owner, minlength=len(segs))
    rows = []
    for k, s in enumerate(segs):
        keep = [i for i in np.argsort(-best[k]) if best[k, i] >= SOUNDS_FLOOR]
        rows.append({"seg": s["seg"], "model": model, "windows": int(n[k]),
                     "p": {labels[i]: round(float(best[k, i]), 3) for i in keep}})
    vocab = sorted(SOUNDS_INLINE["classes"])
    want = [k for k, r in enumerate(rows) if any(r["p"].get(c, 0) >= SOUNDS_FRAME_GATE for c in vocab)]
    chunks = [want[i:i + 8] for i in range(0, len(want), 8)]
    try:
        got = await asyncio.gather(*[ctx.client.sound_frames(
            [x[int(segs[k]["start"] * SR16):int(segs[k]["end"] * SR16)] for k in ch], vocab) for ch in chunks])
    except ValueError as e:                        # a server without the frame models
        print(f"[sounds] {rec['id']}: inline sound timing skipped: {str(e)[:120]}", flush=True)
        got = []
    for ch, (arrs, step, fmodel) in zip(chunks, got):
        for k, a in zip(ch, arrs):
            heard = [c for c in vocab if rows[k]["p"].get(c, 0) >= SOUNDS_FRAME_GATE]
            rows[k]["frames"] = {"model": fmodel, "step_s": step,
                                 **{c: [round(float(v), 3) for v in a[:, vocab.index(c)]] for c in heard}}
    write_jsonl(d / "sounds.jsonl", rows)


# ----------------------------------------------------------------------------- H. fill
def kind_of(text: str) -> str:
    """A voiced gap transcribed alone: a filler, a word the segment's transcript dropped, or nothing."""
    from melati.tags import is_filler
    toks = [t for t in text.replace(",", " ").replace(".", " ").split() if t.strip("-?!")]
    if not toks:
        return "empty"
    return "filler" if all(is_filler(t) for t in toks) else "word"


async def fill(ctx: Ctx, rec, pad: float = 0.10):
    """Each voiced gap between aligned words, padded by 0.1 s of its surroundings and transcribed
    alone. It only works because the ASR model was trained not to write text over voice-free audio:
    a stock Whisper given a 0.3 s breath writes something, and every gap would read as a word."""
    d = ctx.work / rec["id"]
    x, _ = sf.read(str(d / "vocal16k.flac"), dtype="float32")
    segs = {s["seg"]: s for s in read_jsonl(d / "segments.jsonl")}
    tags = read_jsonl(d / "tags.jsonl")

    async def one(s0, g):
        a, b = s0 + g["s"] - pad, s0 + g["e"] + pad
        try:
            r = _summary(await decode(ctx.client, x[max(0, int(a * SR16)):int(b * SR16)]))
            return dict(g, text=r["text"], kind=kind_of(r["text"]))
        except Exception as e:  # noqa: BLE001
            return dict(g, text="", kind="error", error=str(e)[:120])

    # every gap of the recording at once (the client caps what is in flight): one segment at a time, a recording
    # with hundreds of segments waited out the ASR queue hundreds of times over
    jobs = [(t["seg"], g) for t in tags for g in t.get("gaps") or []]
    done = await asyncio.gather(*[one(segs[sid]["start"], g) for sid, g in jobs])
    by = {t["seg"]: [] for t in tags}
    for (sid, _), r in zip(jobs, done):
        by[sid].append(r)
    write_jsonl(d / "fill.jsonl", [{"seg": t["seg"], "gaps": by[t["seg"]]} for t in tags])


# ----------------------------------------------------------------------------- I. export
def export_rows(work: Path, rec, cfg, audio: bool = True, dropped: dict | None = None):
    """Every exported segment of a recording as (manifest row, its 24 kHz audio). The export stage writes them as
    files; scripts/build_hf_dataset.py packs them into the dataset's parquet shards without touching disk.
    audio=False skips decoding (the audio is None) and leaves `lufs` and `peak_db` out. A row the release leaves out
    (melati.stage_export.release_drop) is not returned; `dropped` collects {reason: [rows, seconds]} for them."""
    from collections import Counter
    from melati.stage_export import DEFAULTS, build, load_jsonl, release_drop
    cfg = dict(DEFAULTS, **(cfg or {}))
    d = work / rec["id"]
    segs = read_jsonl(d / "segments.jsonl")
    asr_, al = load_jsonl(d / "asr.jsonl"), load_jsonl(d / "align.jsonl")
    mos, tg, fl = load_jsonl(d / "mos.jsonl"), load_jsonl(d / "tags.jsonl"), load_jsonl(d / "fill.jsonl")
    for seg, a in load_jsonl(d / "audioset.jsonl").items():          # nonverbal/audioset_pass.py, where it has run
        tg[seg] = dict(tg.get(seg) or {}, audioset=a)
    for seg, a in load_jsonl(d / "sounds.jsonl").items():            # the sounds stage, where it has run
        tg[seg] = dict(tg.get(seg) or {}, sounds=a)
    stems = dict(np.load(d / "stems.npz"))
    meta = json.loads((d / "sep.json").read_text())
    said = [r for r in asr_.values() if r.get("text")]
    heard = {k: v / len(said) for k, v in Counter(r.get("detected_language") for r in said).items()} if said else {}
    speech = mix = None
    out = []
    for s in segs:
        if not (asr_.get(s["seg"]) or {}).get("text"):
            continue
        row, (ta, tb), version = build(s, asr_.get(s["seg"]), al.get(s["seg"]), mos.get(s["seg"]),
                                       tg.get(s["seg"]), fl.get(s["seg"]), stems, cfg)
        why = release_drop(row, asr_.get(s["seg"]), heard, cfg)
        if why:
            if dropped is not None:
                dropped.setdefault(why, [0, 0.0])
                dropped[why][0] += 1
                dropped[why][1] += row["duration"]
            continue
        row.update(audio=f"audio/{rec['id']}/{s['seg']}.flac", source_stem=rec.get("stem"))
        if not audio:
            out.append((row, None))
            continue
        if version == "mix" and mix is None:
            mix = A.apply_gain(A.decode(Path(rec["path"]), sr=SR24)[0], meta["gain_db"])
        if version == "vocal" and speech is None:
            speech = sf.read(str(d / "vocal24k.flac"), dtype="float32")[0]
        x = (mix if version == "mix" else speech)[int(ta * SR24):int(tb * SR24)]
        row.update(lufs=round(A.loudness(x, SR24), 2) if len(x) > 0.4 * SR24 else None, peak_db=round(A.peak_db(x), 2))
        out.append((row, x))
    return out


def _export(work: Path, out: Path, rec, cfg):
    rows = []
    for row, x in export_rows(work, rec, cfg):
        A.write_flac(out / row["audio"], x, SR24)
        rows.append(row)
    write_jsonl(work / rec["id"] / "export.jsonl", rows)
    return len(rows)


async def export(ctx: Ctx, rec):
    await ctx.cpu_(_export, ctx.work, ctx.out, rec, ctx.export_cfg)


FUNCS = {"front": front, "diar": diar, "seg": seg, "asr": asr, "align": align, "quality": quality,
         "tag": tag, "sounds": sounds, "fill": fill, "export": export}


async def run_recording(ctx: Ctx, rec, stages=STAGES):
    d = ctx.work / rec["id"]
    for st in stages:
        if (d / OUTPUT[st]).exists():
            continue
        t0 = time.time()
        await FUNCS[st](ctx, rec)
        ctx.timing.setdefault(st, [0.0, 0])
        ctx.timing[st][0] += time.time() - t0
        ctx.timing[st][1] += 1
