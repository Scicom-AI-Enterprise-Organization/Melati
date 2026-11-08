import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from melati import pipeline as P


class FakeSounds:
    """A tagger that hears music in every clip; with frames=False its server has no frame models, as serve/sounds_server
    answers when FRAMES_CKPT does not exist."""
    def __init__(self, frames=True):
        self.has_frames, self.frame_calls = frames, 0

    async def sounds_window(self):
        return 10.24

    async def sounds(self, clips):
        return np.array([[0.9, 0.05]] * len(clips), np.float32), ["Music", "Speech"], "tagger@rev"

    async def sound_frames(self, clips, classes):
        self.frame_calls += 1
        if not self.has_frames:
            raise ValueError("sounds/frames 404: the frame models are not loaded here")
        return [np.full((int(len(x) / P.SR16 / 0.16) + 1, len(classes)), 0.9, np.float32) for x in clips], 0.16, "frames"


def run(tmp_path, client):
    d = tmp_path / "r1"
    d.mkdir()
    sf.write(d / "vocal16k.flac", np.zeros(P.SR16 * 30, np.float32), P.SR16)
    segs = [{"seg": "r1_0000", "start": 0.0, "end": 4.0}, {"seg": "r1_0001", "start": 5.0, "end": 27.0}]
    (d / "segments.jsonl").write_text("".join(json.dumps(s) + "\n" for s in segs))
    with ThreadPoolExecutor(2) as io:
        asyncio.run(P.sounds(P.Ctx(work=tmp_path, client=client, cpu=None, io=io), {"id": "r1"}))
    return P.read_jsonl(d / "sounds.jsonl")


def test_music_is_placed_in_time_where_the_server_has_frame_models(tmp_path):
    rows = run(tmp_path, FakeSounds())
    assert [r["p"] for r in rows] == [{"Music": 0.9, "Speech": 0.05}] * 2
    assert [r["windows"] for r in rows] == [1, 4]          # 22 s in 10.24 s windows every 5.12 s, the last at the end
    assert all(r["frames"]["model"] == "frames" and r["frames"]["Music"] for r in rows)


def test_a_server_without_frame_models_still_gives_every_segment_its_sounds(tmp_path):
    client = FakeSounds(frames=False)
    rows = run(tmp_path, client)
    assert client.frame_calls == 1
    assert [r["p"] for r in rows] == [{"Music": 0.9, "Speech": 0.05}] * 2
    assert not any("frames" in r for r in rows)
