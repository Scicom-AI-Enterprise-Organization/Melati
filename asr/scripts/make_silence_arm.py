#!/usr/bin/env python3
"""Generate the silence / near-silence arm.

Silence is the single best-documented Whisper hallucination trigger: the encoder sees
near-zero input, the decoder falls back to its LM prior, and with
`condition_on_previous_text=True` that output becomes the next window's context and
cascades. Ground truth for every clip here is the empty string, so ANY output is a
hallucination -- which makes this the cleanest arm to read.

Axes: duration (short vs. past the 30 s window boundary), and floor type (true digital
zero, dither, broadband hiss, mains hum, room tone) -- true digital zero behaves
differently from a realistic quiet mic.
"""
import argparse, csv, itertools
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

FLOORS = {
    "digital_zero": lambda n, rng: np.zeros(n),
    "dither":       lambda n, rng: rng.normal(0, 1e-4, n),
    "hiss_-60db":   lambda n, rng: rng.normal(0, 10 ** (-60 / 20), n),
    "hiss_-45db":   lambda n, rng: rng.normal(0, 10 ** (-45 / 20), n),
    "hum_50hz":     lambda n, rng: 10 ** (-50 / 20) * np.sin(2 * np.pi * 50 * np.arange(n) / SR),
    "roomtone":     lambda n, rng: np.convolve(rng.normal(0, 10 ** (-48 / 20), n), np.ones(64) / 64, "same"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "silence")
    ap.add_argument("--durations", nargs="+", type=float,
                    default=[1.0, 5.0, 10.0, 29.0, 31.0, 60.0, 120.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (floor, dur) in enumerate(itertools.product(FLOORS, args.durations)):
        rng = np.random.default_rng(args.seed + i)
        x = np.clip(FLOORS[floor](int(dur * SR), rng), -1, 1).astype(np.float32)
        name = f"{floor}_{dur:g}s.wav"
        sf.write(wav_dir / name, x, SR)
        rows.append({
            "audio_filepath": f"wav/{name}",
            "floor": floor,
            "duration_s": dur,
            # No speech anywhere in this arm: correct output is the empty string.
            "reference_text": "",
        })

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} clips ({sum(r['duration_s'] for r in rows)/60:.1f} min) -> {_disp(wav_dir)}")


if __name__ == "__main__":
    main()
