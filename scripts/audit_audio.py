#!/usr/bin/env python3
"""What did the export do to the audio? Bandwidth, level and clipping of a set of segment files.

Two Emilia-Pipe export defects are measured here, on any set of files (a directory or a zip):

  * Amphion issue #436: `write_mp3` uses pydub's default MP3 export, which ffmpeg encodes at
    32 kbit/s, and LAME at that rate low-passes "24 kHz" audio at about 8 kHz. We read each
    file's own bitrate and find its spectral cutoff: the highest frequency at which the
    long-term average spectrum is still within 50 dB of its 1-4 kHz level.
  * per-segment peak normalisation: `x / max|x| * 32767` puts every segment's loudest sample at
    full scale, so segment loudness follows the loudest transient. We report the spread of
    integrated loudness and of peak level across files, and the share of files with samples at
    or above -0.1 dBFS.

    python scripts/audit_audio.py --name emilia_v1 --zip .../malaysian-podcast-processed.zip \\
        --sample 2000 --out results/emilia/audit_emilia_v1.json
"""
import argparse, io, json, random, subprocess, zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")


def ffprobe(data: bytes):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,bit_rate,codec_name",
                          "-show_entries", "format=bit_rate", "-of", "json", "pipe:0"],
                         input=data, capture_output=True).stdout
    try:
        d = json.loads(out)
        st = (d.get("streams") or [{}])[0]
        br = st.get("bit_rate") or (d.get("format") or {}).get("bit_rate")
        return st.get("codec_name"), int(st.get("sample_rate") or 0), int(br) if br else None
    except Exception:
        return None, 0, None


def decode(data: bytes):
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return x.mean(1), sr
    except Exception:
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "pipe:1"],
                             input=data, capture_output=True).stdout
        _, sr, _ = ffprobe(data)
        return np.frombuffer(out, np.float32), sr


def ltas_db(x, sr, n_fft=2048):
    if len(x) < n_fft:
        return None, None
    hop = n_fft // 2
    w = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * w
    p = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    return np.fft.rfftfreq(n_fft, 1 / sr), 10 * np.log10(p + 1e-20)


def cutoff_hz(freqs, db, ref_band=(1000, 4000), drop=50.0):
    ref = np.median(db[(freqs >= ref_band[0]) & (freqs <= ref_band[1])])
    above = np.flatnonzero(db >= ref - drop)
    return float(freqs[above[-1]]) if len(above) else 0.0


def loudness(x, sr):
    import pyloudnorm as pyln
    if len(x) < int(0.4 * sr) or not np.any(x):
        return None
    v = pyln.Meter(sr).integrated_loudness(x.astype(np.float64))
    return float(v) if np.isfinite(v) else None


def items(args):
    if args.zip:
        z = zipfile.ZipFile(args.zip)
        names = [n for n in z.namelist() if n.lower().endswith(AUDIO)]
        random.Random(0).shuffle(names)
        for n in names[: args.sample]:
            yield n, z.read(n)
    else:
        files = sorted(p for p in Path(args.dir).rglob("*") if p.suffix.lower() in AUDIO)
        random.Random(0).shuffle(files)
        for p in files[: args.sample]:
            yield str(p), p.read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip")
    g.add_argument("--dir")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows, ltas_sum, ltas_sr, n_ltas = [], None, None, 0
    for name, data in items(args):
        codec, sr_hdr, br = ffprobe(data)
        x, sr = decode(data)
        if len(x) < 2048:
            continue
        f, db = ltas_db(x, sr)
        peak = float(np.max(np.abs(x)))
        rows.append({"codec": codec, "sr": sr, "bitrate": br, "dur": len(x) / sr,
                     "cutoff_hz": cutoff_hz(f, db), "lufs": loudness(x, sr),
                     "peak_db": 20 * np.log10(peak) if peak > 0 else None,
                     "near_full_scale": bool(peak >= 10 ** (-0.1 / 20))})
        if ltas_sr in (None, sr):
            ltas_sr = sr
            ltas_sum = db if ltas_sum is None else ltas_sum + db
            n_ltas += 1
    def pct(v, q):
        v = [a for a in v if a is not None]
        return round(float(np.percentile(v, q)), 3) if v else None
    br = [r["bitrate"] for r in rows if r["bitrate"]]
    out = {
        "name": args.name, "files": len(rows),
        "codecs": {c: sum(r["codec"] == c for r in rows) for c in {r["codec"] for r in rows}},
        "sample_rates": {str(s): sum(r["sr"] == s for r in rows) for s in {r["sr"] for r in rows}},
        "bitrate_kbps": {"p10": pct([b / 1000 for b in br], 10), "p50": pct([b / 1000 for b in br], 50),
                         "p90": pct([b / 1000 for b in br], 90)},
        "cutoff_khz": {q: pct([r["cutoff_hz"] / 1000 for r in rows], q) for q in (10, 50, 90)},
        "share_cutoff_below_9khz": round(float(np.mean([r["cutoff_hz"] < 9000 for r in rows])), 4) if rows else None,
        "lufs": {q: pct([r["lufs"] for r in rows], q) for q in (5, 25, 50, 75, 95)},
        "lufs_iqr": (pct([r["lufs"] for r in rows], 75) or 0) - (pct([r["lufs"] for r in rows], 25) or 0),
        "peak_db": {q: pct([r["peak_db"] for r in rows], q) for q in (5, 50, 95)},
        "share_peak_at_full_scale": round(float(np.mean([r["near_full_scale"] for r in rows])), 4) if rows else None,
        "ltas_sr": ltas_sr,
        "ltas_db": [round(float(v), 2) for v in (ltas_sum / max(1, n_ltas))] if ltas_sum is not None else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "ltas_db"}, indent=1))


if __name__ == "__main__":
    main()
