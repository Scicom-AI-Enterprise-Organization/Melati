#!/usr/bin/env python3
"""Generate the reduplication arm: N repeats of a CV syllable, e.g. "tu-tu-tu-tu".

Motivating failure (2026-09-15): a self-recorded clip saying "it sounds like
tutututu" with the syllable repeated exactly FOUR times decoded as `tu` repeated until
the max-token limit. Whisper's decoder, once inside a repeated-token region, has no
signal telling it how many repeats remain, so it can run away.

The arm sweeps the axes that plausibly control that runaway so a probe can find the
cliff edge rather than just confirm one anecdote:

  * syllable        which CV unit is reduplicated
  * n_repeats       3..12 -- how many are ACTUALLY there (the ground truth)
  * rate            syllables/sec; faster = fewer encoder frames per syllable
  * tail_silence    trailing silence, which independently triggers looping
  * carrier         bare burst vs. embedded in a spoken-like carrier gap

Synthesis is source-filter (pulse train + formant resonators) in numpy -- no TTS/STT
endpoint is touched, so this is safe to run on the laptop. For a natural-voice version
of the same grid, render arms/reduplication_sentences.txt through TTS ON THE GPU BOX.
"""
import argparse, csv, itertools, math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

def _disp(p: Path) -> str:
    """Path for display; falls back to the absolute path when it is outside ROOT."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

SR = 16000

# (F1, F2, F3) in Hz for the vowel nucleus.
VOWELS = {
    "u": (320, 800, 2500),
    "a": (700, 1220, 2600),
    "i": (270, 2290, 3010),
    "e": (530, 1840, 2480),
    "o": (570, 840, 2410),
}
# (burst centre Hz, burst gain, voiced onset) for the consonant.
CONSONANTS = {
    "t": (4000, 0.55, False),
    "k": (2000, 0.50, False),
    "p": (1000, 0.45, False),
    "d": (3200, 0.35, True),
    "b": (900, 0.30, True),
    "g": (1800, 0.32, True),
    "l": (1400, 0.10, True),
    "n": (1000, 0.10, True),
    "m": (700, 0.10, True),
}


def resonator(x: np.ndarray, freq: float, bw: float, sr: int = SR) -> np.ndarray:
    """Two-pole resonator (Klatt-style formant filter)."""
    r = math.exp(-math.pi * bw / sr)
    theta = 2 * math.pi * freq / sr
    a1, a2 = 2 * r * math.cos(theta), -(r * r)
    b0 = (1 - r) * math.sqrt(1 - 2 * r * math.cos(2 * theta) + r * r)
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b0 * xi + a1 * y1 + a2 * y2
        y[i], y2, y1 = yi, y1, yi
    return y


def glottal_pulse_train(n: int, f0: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Impulse train at f0 with a little period jitter, so it isn't robotically periodic."""
    src = np.zeros(n)
    t, period = 0.0, SR / f0
    while t < n:
        src[int(t)] = 1.0
        t += period * (1.0 + rng.normal(0, jitter))
    return src


def syllable(cons: str, vow: str, dur: float, f0: float, rng: np.random.Generator) -> np.ndarray:
    """One CV syllable: consonant burst + formant-filtered voiced nucleus."""
    n = int(dur * SR)
    burst_f, burst_gain, voiced_onset = CONSONANTS[cons]
    f1, f2, f3 = VOWELS[vow]

    # Consonant release burst: short filtered noise transient.
    nb = int(0.012 * SR)
    burst = rng.normal(0, 1, nb) * np.exp(-np.linspace(0, 6, nb))
    burst = resonator(burst, burst_f, 900) * burst_gain

    # Voiced nucleus.
    nv = n - nb
    src = glottal_pulse_train(nv, f0, 0.02, rng)
    src += rng.normal(0, 0.01, nv)  # aspiration floor
    voiced = 1.0 * resonator(src, f1, 70) + 0.45 * resonator(src, f2, 110) + 0.18 * resonator(src, f3, 170)

    # Amplitude envelope: quick attack, gentle decay, short release.
    env = np.ones(nv)
    atk, rel = int(0.015 * SR), int(0.035 * SR)
    env[:atk] = np.linspace(0, 1, atk)
    env[-rel:] = np.linspace(1, 0, rel)
    env *= np.exp(-np.linspace(0, 0.8, nv))
    voiced *= env

    # Normalise the nucleus first, then set the burst RELATIVE to it. Normalising the
    # concatenation instead lets the plosive transient (which is several times taller than
    # the vowel) dominate, leaving an audible click + a squashed vowel.
    vpeak = np.abs(voiced).max()
    voiced = voiced / vpeak * 0.8 if vpeak > 0 else voiced
    bpeak = np.abs(burst).max()
    burst = burst / bpeak * (0.8 * burst_gain) if bpeak > 0 else burst

    out = np.concatenate([burst, voiced])
    if voiced_onset:  # voice bar through the closure for /b d g l m n/
        out[:nb] += 0.12 * resonator(glottal_pulse_train(nb, f0, 0.02, rng), 200, 60)
    return np.clip(out, -1.0, 1.0)


def unit_cv(label, dur, f0, rng):
    """Consonant+vowel, e.g. "tu" -> the tutututu case."""
    return syllable(label[0], label[1], dur, f0, rng)


def unit_vowel(label, dur, f0, rng):
    """Bare sustained vowel, e.g. "a a a a" -- reduplication with no consonant landmark
    at all, so the decoder has even less to segment on."""
    f1, f2, f3 = VOWELS[label]
    n = int(dur * SR)
    src = glottal_pulse_train(n, f0, 0.02, rng) + rng.normal(0, 0.01, n)
    y = 1.0 * resonator(src, f1, 70) + 0.45 * resonator(src, f2, 110) + 0.18 * resonator(src, f3, 170)
    env = np.ones(n)
    edge = int(0.02 * SR)
    env[:edge], env[-edge:] = np.linspace(0, 1, edge), np.linspace(1, 0, edge)
    y *= env
    peak = np.abs(y).max()
    return y / peak * 0.8 if peak > 0 else y


def unit_laugh(label, dur, f0, rng):
    """Breathy /h/ onset + vowel -- the "hahaha" / "hehehe" family."""
    nh = int(min(0.05, dur * 0.35) * SR)
    asp = resonator(rng.normal(0, 1, nh), 1600, 1200) * np.linspace(0.4, 1.0, nh)
    vow = unit_vowel(label, dur - nh / SR, f0 * rng.uniform(0.95, 1.08), rng)
    hpeak = np.abs(asp).max()
    asp = asp / hpeak * 0.8 * 0.3 if hpeak > 0 else asp
    return np.concatenate([asp, vow])


def unit_click(label, dur, f0, rng):
    """Non-speech periodic tick/beep. Repetitive, speech-adjacent in rate, zero lexical
    content -- separates 'repetition' from 'speech' as the trigger."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    if label == "beep":
        y = np.sin(2 * np.pi * 900 * t) * np.exp(-t * 18)
    else:  # "tick" -- damped 2.6 kHz click, ~15 ms so it survives mel framing
        y = np.sin(2 * np.pi * 2600 * t) * np.exp(-t * 220)
    peak = np.abs(y).max()
    return y / peak * 0.8 if peak > 0 else y


# pattern name -> (unit builder, default labels, how the label renders in the reference text)
PATTERNS = {
    "cv":     (unit_cv,    ["tu", "ta", "ka", "pa", "da", "la", "na", "bi", "ko", "me"], lambda l: l),
    "vowel":  (unit_vowel, ["a", "i", "u", "e", "o"],                                    lambda l: l),
    "laugh":  (unit_laugh, ["a", "e", "i"],                                              lambda l: "ha" if l == "a" else f"h{l}"),
    "click":  (unit_click, ["tick", "beep"],                                             lambda l: f"[{l}]"),
}


def build_clip(unit_fn, label, n_repeats, rate, tail_silence, f0, seed):
    rng = np.random.default_rng(seed)
    period = 1.0 / rate
    dur = period * 0.62  # duty cycle: unit vs. inter-unit gap
    gap = np.zeros(int((period - dur) * SR))
    body = np.concatenate([np.concatenate([unit_fn(label, dur, f0, rng), gap]) for _ in range(n_repeats)])
    lead = np.zeros(int(0.25 * SR))
    tail = np.zeros(int(tail_silence * SR))
    clip = np.concatenate([lead, body, tail])
    clip += rng.normal(0, 1e-4, clip.shape)  # dither; digital-silence tails are their own trigger
    return np.clip(clip, -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "reduplication")
    ap.add_argument("--patterns", nargs="+", default=list(PATTERNS), choices=list(PATTERNS))
    ap.add_argument("--labels", nargs="+", default=None, help="override the unit list for every pattern")
    ap.add_argument("--repeats", nargs="+", type=int, default=[3, 4, 5, 6, 8, 12])
    ap.add_argument("--rates", nargs="+", type=float, default=[3.0, 5.0, 7.0, 9.0])
    ap.add_argument("--tails", nargs="+", type=float, default=[0.0, 2.0, 8.0])
    ap.add_argument("--f0", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows, i = [], 0

    for pattern in args.patterns:
        unit_fn, default_labels, render = PATTERNS[pattern]
        for label, n, rate, tail in itertools.product(
            args.labels or default_labels, args.repeats, args.rates, args.tails
        ):
            clip = build_clip(unit_fn, label, n, rate, tail, args.f0, args.seed + i)
            i += 1
            name = f"{pattern}_{label}_n{n:02d}_r{rate:g}_t{tail:g}.wav"
            sf.write(wav_dir / name, clip, SR)
            token = render(label)
            rows.append(
                {
                    "audio_filepath": f"wav/{name}",
                    "pattern": pattern,
                    "unit": label,
                    "n_repeats": n,
                    "rate_hz": rate,
                    "tail_silence_s": tail,
                    "duration_s": round(len(clip) / SR, 3),
                    # Ground truth: exactly n repeats. More than n in a hypothesis is a loop.
                    "reference_text": " ".join([token] * n),
                    "reference_collapsed": token * n,
                }
            )

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total = sum(r["duration_s"] for r in rows)
    by_pat = Counter(r["pattern"] for r in rows)
    print(f"wrote {len(rows)} clips ({total/60:.1f} min) -> {_disp(wav_dir)}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_pat.items())))
    print(f"manifest -> {_disp(manifest)}")


if __name__ == "__main__":
    main()
