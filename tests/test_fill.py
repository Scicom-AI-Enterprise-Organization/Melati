import asyncio
import json

import numpy as np
import soundfile as sf

from melati import pipeline as P


class FakeASR:
    def __init__(self):
        self.calls = 0

    async def transcribe(self, x, language=None, temperature=0.0, max_tokens=None, seed=None):
        self.calls += 1
        await asyncio.sleep(0)
        return {"text": "um", "segments": [], "language": "ms"}


def test_fill_sends_every_gap_and_keeps_each_in_its_segment(tmp_path):
    d = tmp_path / "r1"
    d.mkdir()
    sf.write(d / "vocal16k.flac", np.zeros(16000 * 10, np.float32), 16000)
    segs = [{"seg": "r1_0000", "start": 0.0, "end": 4.0}, {"seg": "r1_0001", "start": 5.0, "end": 9.0},
            {"seg": "r1_0002", "start": 9.0, "end": 9.8}]
    tags = [{"seg": "r1_0000", "gaps": [{"s": 1.0, "e": 1.4}, {"s": 2.0, "e": 2.3}]},
            {"seg": "r1_0001", "gaps": [{"s": 0.5, "e": 0.9}]}, {"seg": "r1_0002"}]
    (d / "segments.jsonl").write_text("".join(json.dumps(s) + "\n" for s in segs))
    (d / "tags.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tags))
    asr = FakeASR()
    asyncio.run(P.fill(P.Ctx(work=tmp_path, client=asr, cpu=None, io=None), {"id": "r1"}))
    rows = P.read_jsonl(d / "fill.jsonl")
    assert asr.calls == 3
    assert [r["seg"] for r in rows] == ["r1_0000", "r1_0001", "r1_0002"]
    assert [[g["s"] for g in r["gaps"]] for r in rows] == [[1.0, 2.0], [0.5], []]
    assert all(g["kind"] == "filler" for r in rows for g in r["gaps"])
