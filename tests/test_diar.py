import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from melati import pipeline as P


class FakeDiar:
    """A one pass over ten minutes never answers; anything shorter does, with one speaker throughout."""
    def __init__(self):
        self.calls = []

    async def diarize(self, path, duration_s, tries=5):
        self.calls.append((round(duration_s), tries))
        if duration_s > 600:
            raise RuntimeError("diar/v1/audio/transcriptions failed after 2 tries: TimeoutError: ")
        return f"[0.0][S01][{duration_s:.1f}]", False


def diarize(tmp_path, seconds):
    d = tmp_path / "r1"
    d.mkdir()
    sf.write(d / "speech16k.flac", np.zeros(16000 * seconds, np.float32), 16000)
    dz = FakeDiar()
    with ThreadPoolExecutor(2) as io:
        asyncio.run(P.diar(P.Ctx(work=tmp_path, client=dz, cpu=None, io=io), {"id": "r1"}))
    return dz.calls, json.loads((d / "diar.json").read_text())


def test_a_one_pass_that_answers_keeps_speakers_across_the_recording(tmp_path):
    calls, out = diarize(tmp_path, 60)
    assert calls == [(60, 2)]
    assert out["mode"] == "one" and out["turns"] == [[0.0, 60.0, "S01"]] and "one_pass_error" not in out


def test_a_one_pass_that_never_answers_is_diarized_in_windows(tmp_path):
    calls, out = diarize(tmp_path, 900)
    assert calls == [(900, 2), (600, 5), (600, 5)]
    assert out["mode"] == "windows" and out["one_pass_error"].startswith("diar/v1/audio/transcriptions failed")
    assert out["turns"] == [[0.0, 450.0, "w0S01"], [450.0, 900.0, "w1S01"]]


def test_a_recording_over_the_one_pass_limit_goes_straight_to_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(P.DZ, "ONE_PASS_MAX_S", 600)
    calls, out = diarize(tmp_path, 900)
    assert calls == [(600, 5), (600, 5)]
    assert out["mode"] == "windows" and "one_pass_error" not in out
