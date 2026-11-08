"""The loop guard on Whisper (melati.pipeline.decode): Whisper's own temperature fallback, on vLLM."""
import asyncio

import numpy as np

from melati.pipeline import LOOP_GZIP, SR16, decode, gzip_ratio

LOOP = " ".join(["ta"] * 37)          # what a replay fine-tune wrote for "ta" said 8 times


class FakeASR:
    """Answers with a loop at the temperatures listed in `loops_at`, a sentence otherwise; records calls."""

    def __init__(self, loops_at):
        self.loops_at, self.calls = set(loops_at), []

    async def transcribe(self, x, language=None, temperature=0.0, max_tokens=None, seed=None):
        self.calls.append({"t": temperature, "cap": max_tokens, "seed": seed})
        return {"text": LOOP if temperature in self.loops_at else "ta ta ta ta ta ta ta ta", "language": "ms"}


def run(client, seconds=3.0):
    return asyncio.run(decode(client, np.zeros(int(seconds * SR16), np.float32)))


def test_gzip_ratio_separates_a_loop_from_real_repetition():
    assert gzip_ratio(LOOP) > LOOP_GZIP
    assert gzip_ratio("ha ha ha ha ha") < LOOP_GZIP          # laughter said five times is not a loop
    assert gzip_ratio("") == 0.0


def test_greedy_kept_when_it_does_not_loop():
    c = FakeASR(loops_at=())
    r = run(c)
    assert r["fallback_t"] == 0.0 and len(c.calls) == 1 and c.calls[0]["t"] == 0.0


def test_loop_falls_back_to_the_first_temperature_that_does_not_loop():
    c = FakeASR(loops_at=(0.0, 0.2))
    r = run(c)
    assert r["fallback_t"] == 0.4 and r["text"] != LOOP and r["gzip_ratio"] <= LOOP_GZIP
    assert [k["t"] for k in c.calls] == [0.0, 0.2, 0.4]
    assert all(k["seed"] == 0 for k in c.calls[1:])           # a fallback is reproducible


def test_every_decode_is_capped_by_the_clip_length():
    c = FakeASR(loops_at=(0.0, 0.2, 0.4, 0.6))
    r = run(c, seconds=3.0)
    assert r["gzip_ratio"] > LOOP_GZIP                         # still a loop: kept, and flagged downstream
    assert {k["cap"] for k in c.calls} == {32 + 45}            # 32 + 15 tokens/s x 3 s, on every attempt
