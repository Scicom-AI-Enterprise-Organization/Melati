import asyncio
import time

import numpy as np

from serve.batching import DynamicBatcher, dec_f16, dec_pcm16, enc_f16, enc_pcm16


def submit(fn, *requests, **kw):
    """Each request (a list of items) submitted after the one before has its answer -> (batcher, answers)."""
    async def go():
        b = DynamicBatcher(fn, **kw)
        b.start()
        return b, [await asyncio.wait_for(b.submit_many(items, raise_errors=False), 10) for items in requests]
    return asyncio.run(go())


def test_items_that_queue_while_the_model_runs_share_one_batch():
    sizes = []

    def fn(batch):
        sizes.append(len(batch))
        time.sleep(0.05)
        return [x * 2 for x in batch]
    b, (res,) = submit(fn, list(range(10)), max_batch=4, max_wait_ms=20)
    assert res == [x * 2 for x in range(10)]
    assert sizes == [4, 4, 2] and b.report()["mean_batch"] == 3.33


def test_a_batch_that_fails_fails_only_its_own_requests():
    def fn(batch):
        if "bad" in batch:
            raise ValueError("bad item")
        return [s.upper() for s in batch]
    b, (first, second) = submit(fn, ["a", "bad"], ["c", "d"], max_batch=2, max_wait_ms=1)
    assert all(isinstance(r, ValueError) for r in first) and second == ["C", "D"]
    assert b.stats["errors"] == 2


def test_a_model_that_drops_an_item_fails_the_batch_instead_of_leaving_it_waiting():
    _, (res,) = submit(lambda batch: batch[:-1], [1, 2, 3], max_batch=8, max_wait_ms=1)
    assert all(isinstance(r, RuntimeError) for r in res)


def test_audio_survives_the_json_codec():
    x = (0.5 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    assert np.abs(dec_pcm16(enc_pcm16(x)) - x).max() <= 1 / 32767
    assert dec_pcm16(enc_pcm16(np.array([2.0, -2.0], np.float32))).tolist() == [1.0, -1.0]
    y = np.array([0.1, 1000.0, -3.5], np.float32)
    assert np.allclose(dec_f16(enc_f16(y)), y, rtol=1e-3)
