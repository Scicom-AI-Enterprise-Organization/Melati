"""Audio I/O for every stage: decode anything, resample, measure loudness, write FLAC.

Two Emilia-Pipe defects shape this module, and neither may come back (docs/emilia_bugs.md):

* Emilia writes every segment with pydub's default MP3 export, which ffmpeg encodes at 32 kbit/s
  and low-passes at about 8 kHz (Amphion issue #436). Melati writes FLAC only. There is no lossy
  encoder anywhere between the source and the dataset.
* Emilia peak-normalises twice: the whole file after a loudness gain clamped to +-3 dB (which
  undoes the loudness normalisation), and then every segment to full scale on export (so each
  segment's level depends on its loudest transient). Melati applies ONE gain per recording, a
  loudness target with a peak ceiling, so a whisper stays quieter than a shout from the same
  speaker. Each segment's own loudness is measured and stored, never imposed.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_LUFS = -23.0      # EBU R128 programme loudness, one gain per recording
PEAK_CEILING_DB = -1.0   # never push a sample above this, whatever the loudness target asks


def decode(path_or_bytes, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """float32 audio and its rate. libsndfile first (WAV/FLAC/OGG/MP3), ffmpeg for the rest
    (M4A, WebM/Opus from YouTube). `sr` resamples; None keeps the native rate."""
    data = path_or_bytes.read_bytes() if isinstance(path_or_bytes, Path) else path_or_bytes
    try:
        src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else str(data)
        x, rate = sf.read(src, dtype="float32", always_2d=True)
    except Exception:
        x, rate = _ffmpeg(data)
    if mono:
        x = x.mean(axis=1)
    if sr and rate != sr:
        x, rate = resample(x, rate, sr), sr
    return np.ascontiguousarray(x, dtype=np.float32), int(rate)


def _ffmpeg(data) -> tuple[np.ndarray, int]:
    args = ["ffmpeg", "-nostdin", "-v", "error"]
    if isinstance(data, (bytes, bytearray)):
        args += ["-i", "pipe:0"]
        inp = bytes(data)
    else:
        args += ["-i", str(data)]
        inp = None
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=sample_rate,channels", "-of", "csv=p=0",
                            "pipe:0" if inp is not None else str(data)],
                           input=inp, capture_output=True, check=True).stdout.decode().strip().split(",")
    rate, ch = int(probe[0]), int(probe[1])
    out = subprocess.run(args + ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
                         input=inp, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, ch).copy(), rate


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr
    return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def loudness(x: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770-4, gated). -inf for silence or < 0.4 s."""
    import pyloudnorm as pyln
    if len(x) < int(0.4 * sr) or not np.any(x):
        return float("-inf")
    return float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))


def peak_db(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return 20 * np.log10(p) if p > 0 else float("-inf")


def recording_gain_db(x: np.ndarray, sr: int, target: float = TARGET_LUFS,
                      ceiling: float = PEAK_CEILING_DB) -> float:
    """The one gain applied to a whole recording: reach `target` LUFS unless that would push
    the peak over `ceiling`, in which case stop at the ceiling. Quiet recordings stay a little
    quiet rather than clipping; nothing is limited or compressed."""
    lufs = loudness(x, sr)
    if not np.isfinite(lufs):
        return 0.0
    return float(min(target - lufs, ceiling - peak_db(x)))


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * (10 ** (gain_db / 20))).astype(np.float32)


def write_flac(path: Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Atomic: written to a sibling and renamed, so a killed worker never leaves a file that
    looks finished."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    sf.write(tmp, np.clip(x, -1.0, 1.0), sr, format="FLAC", subtype=subtype)
    tmp.replace(path)


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * np.log10(r) if r > 0 else float("-inf")


def frame_energy_db(x: np.ndarray, sr: int, frame_s: float = 0.02) -> np.ndarray:
    """Per-frame RMS in dB, the coarse energy track the gap and edge checks read."""
    n = max(1, int(frame_s * sr))
    m = len(x) // n
    if m == 0:
        return np.full(1, rms_db(x))
    f = x[: m * n].reshape(m, n).astype(np.float64)
    r = np.sqrt(np.mean(f * f, axis=1))
    return 20 * np.log10(np.maximum(r, 1e-10))
