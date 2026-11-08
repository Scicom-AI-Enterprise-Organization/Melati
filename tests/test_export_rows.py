import json
import sys

import numpy as np
import pytest
import soundfile as sf

from melati import pipeline as P
from melati import stage_export as SE


def work_dir(tmp_path):
    """One recording through every stage: a segment in English, and one the ASR model heard as Arabic and wrote as
    Tamil, which the release leaves out (melati.stage_export.release_drop)."""
    d = tmp_path / "work" / "r1"
    d.mkdir(parents=True)
    t = np.arange(P.SR24 * 6) / P.SR24
    voice = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    src = tmp_path / "episode.flac"
    sf.write(src, voice, P.SR24)
    sf.write(d / "vocal24k.flac", voice, P.SR24)
    n = int(6 / 0.05)
    np.savez(d / "stems.npz", frame_s=0.05, mix=np.full(n, -20, np.float16), speech=np.full(n, -20, np.float16),
             music=np.full(n, -80, np.float16), sfx=np.full(n, -80, np.float16))
    (d / "sep.json").write_text(json.dumps({"gain_db": 0.0, "duration_s": 6.0}))
    segs = ("r1_0000", "r1_0001")
    stages = {
        "segments": [{"seg": "r1_0000", "rec": "r1", "speaker": "S01", "start": 0.0, "end": 3.0},
                     {"seg": "r1_0001", "rec": "r1", "speaker": "S01", "start": 3.5, "end": 6.0}],
        "asr": [{"seg": "r1_0000", "text": "hello there", "language": "en", "detected_language": "en"},
                {"seg": "r1_0001", "text": "vanakkam", "language": "ta", "detected_language": "ar"}],
        "align": [{"seg": "r1_0000", "words": [{"w": "hello", "s": 0.5, "e": 0.9, "c": 0.9},
                                               {"w": "there", "s": 1.0, "e": 1.4, "c": 0.9}], "score": 0.9,
                   "tail_s": 1.6},
                  {"seg": "r1_0001", "words": [{"w": "vanakkam", "s": 0.3, "e": 1.0, "c": 0.9}], "score": 0.9,
                   "tail_s": 1.5}],
        "mos": [{"seg": s, "mos_vocal": 3.5, "mos_mix": 3.6, "dnsmos_vocal": {"OVRL": 3.0}, "dnsmos_mix": {"OVRL": 3.1}}
                for s in segs],
        "tags": [{"seg": s} for s in segs],
        "fill": [{"seg": s, "gaps": []} for s in segs],
        "sounds": [{"seg": "r1_0000", "model": "tagger@rev", "windows": 1, "p": {"Speech": 0.95, "Music": 0.05}}],
    }
    for name, rows in stages.items():
        P.write_jsonl(d / f"{name}.jsonl", rows)
    return tmp_path / "work", {"id": "r1", "path": str(src), "stem": "episode"}


def test_the_export_leaves_out_what_the_release_does_and_keeps_the_sounds(tmp_path):
    work, rec = work_dir(tmp_path)
    dropped = {}
    out = P.export_rows(work, rec, {}, dropped=dropped)
    assert [row["id"] for row, _ in out] == ["r1_0000"]
    assert list(dropped) == ["language_fallback"] and dropped["language_fallback"][0] == 1
    row, x = out[0]
    assert row["text"] == "hello there" and row["sounds"] == [{"label": "Speech", "p": 0.95}]
    assert row["audio"] == "audio/r1/r1_0000.flac" and row["source_stem"] == "episode"
    assert row["audio_version"] == "mix" and abs(len(x) - row["duration"] * P.SR24) <= 1


def test_the_export_command_writes_the_rows_the_dataset_holds(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    work, rec = work_dir(tmp_path)
    (tmp_path / "sources.jsonl").write_text(json.dumps(rec) + "\n")
    out = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", ["stage_export", "--sources", str(tmp_path / "sources.jsonl"),
                                      "--work", str(work), "--out", str(out)])
    SE.main()
    rows = pq.read_table(out / "manifest" / "part-000.parquet").to_pylist()
    assert rows == [row for row, _ in P.export_rows(work, rec, {})]
    assert sf.info(str(out / rows[0]["audio"])).samplerate == P.SR24
