import asyncio
import json
import os

from melati.client import Client, Endpoints


def write(p, d, t):
    p.write_text(json.dumps(d))
    os.utime(p, (t, t))


class Reply:
    def __init__(self, status, body):
        self.status, self.body = status, body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self.body

    async def text(self):
        return json.dumps(self.body)


class Session:
    """Answers each post with the next (status, body) in turn and keeps the urls posted to."""
    def __init__(self, replies):
        self.replies, self.urls = list(replies), []

    def post(self, url, json=None, data=None):
        self.urls.append(url)
        return Reply(*self.replies.pop(0))


def test_endpoints_pick_up_new_replicas_and_keep_a_service_that_vanished(tmp_path):
    p = tmp_path / "endpoints.json"
    write(p, {"front": ["http://a"], "tag": ["http://t1"]}, 1000)
    ep = Endpoints(p)
    ep.RELOAD_S = 0
    assert ep.next("front") == "http://a"
    write(p, {"front": ["http://a", "http://b"]}, 2000)        # a replica added; tag momentarily unhealthy
    assert {ep.next("front") for _ in range(4)} == {"http://a", "http://b"}
    assert ep.next("tag") == "http://t1"


def test_endpoints_drop_a_replica_that_left(tmp_path):
    p = tmp_path / "endpoints.json"
    write(p, {"tag": ["http://t1", "http://t2"]}, 1000)
    ep = Endpoints(p)
    ep.RELOAD_S = 0
    write(p, {"tag": ["http://t2"]}, 2000)
    assert {ep.next("tag") for _ in range(4)} == {"http://t2"}


def test_a_call_goes_to_the_replica_with_the_fewest_open(tmp_path):
    p = tmp_path / "endpoints.json"
    write(p, {"diar": ["http://d0", "http://d1", "http://d2"]}, 1000)
    ep = Endpoints(p)
    ep.inflight.update({"http://d0": 5, "http://d1": 1, "http://d2": 3})
    assert {ep.next("diar") for _ in range(6)} == {"http://d1"}


def test_a_replica_that_failed_is_skipped_while_another_is_up(tmp_path):
    p = tmp_path / "endpoints.json"
    write(p, {"tag": ["http://t0", "http://t1"]}, 1000)
    ep = Endpoints(p)
    ep.failed("http://t0")
    assert {ep.next("tag") for _ in range(4)} == {"http://t1"}
    ep.failed("http://t1")                                    # all down: keep trying them all
    assert {ep.next("tag") for _ in range(4)} == {"http://t0", "http://t1"}


def test_an_error_sent_as_200_is_retried_unless_it_is_an_empty_decode(tmp_path):
    p = tmp_path / "endpoints.json"
    write(p, {"asr": ["http://a0", "http://a1"]}, 1000)
    c = Client(Endpoints(p))
    path = "/v1/audio/transcriptions"

    async def run():
        c.session = Session([(200, {"error": {"message": "tuple index out of range", "code": 500}})])
        assert await c._post("asr", path) == {"text": "", "segments": []}
        assert c.ep.cool == {}                                   # an empty transcript is an answer
        c.session = Session([(200, {"error": {"message": "EngineCore encountered an issue", "code": 500}}),
                             (200, {"text": "Selamat pagi."})])
        assert (await c._post("asr", path))["text"] == "Selamat pagi."
        assert len(set(c.session.urls)) == 2                     # the failing replica is set aside, not trusted

    asyncio.run(run())
