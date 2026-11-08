"""The pipeline's only door to a GPU: async calls to the model servers listed in endpoints.json.

serve/launch.py writes the endpoints file ({service: [url, ...]}) from serve/cluster.json; every
call here goes to the replica of its service with the fewest calls open, retries on a connection
error or a 5xx with backoff on another, and caps how many requests each service has in flight
(`limits`), so adding a replica is the whole of scaling a model: no client code changes.

    ep = Endpoints("$MELATI_DATA/serve/endpoints.json")
    async with Client(ep) as c:
        tracks = await c.sed(x32k)
        stems = await c.separate(x44k)
        resp = await c.transcribe(x16k)               # vLLM Whisper, OpenAI-compatible
        words = await c.align([(x16k, text, "ms")])
"""
from __future__ import annotations

import asyncio
import base64
import io
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from serve.batching import dec_f16, enc_pcm16

DEFAULT_LIMITS = {"asr": 512, "diar": 64, "front": 32, "align": 256, "mos": 256, "tag": 256, "uvr": 4, "omni": 128,
                  "sounds": 256}


def empty_decode(err) -> bool:
    """vLLM 0.17's verbose_json answer to a decode that is only end-of-text, i.e. the model heard nothing to write:
    SpeechToText._get_verbose_segments reads the second-to-last token of a one-token tuple and the route sends the
    IndexError back as an error body. It is an empty transcript; any other error body is a failure, retried like
    a 5xx, where until 2026-10-26 it was read as one too."""
    return isinstance(err, dict) and err.get("message") == "tuple index out of range"


class Endpoints:
    RELOAD_S = 30
    COOL_S = 30

    def __init__(self, path):
        """One endpoints file, or several (services of separate clusters), merged. Re-read when one changes
        (checked every RELOAD_S), so replicas added by `launch.py up` mid-run take work without a restart.

        A call goes to the replica with the fewest calls this process has open there (ties in turn), and a replica
        that just failed one is skipped for COOL_S while another is up. Round-robin alone let one diarization
        server queue a hundred whole recordings behind two long ones while its sibling had room."""
        self.paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
        self.path = ", ".join(map(str, self.paths))
        self.inflight = defaultdict(int)
        self.cool = {}
        self._turn = itertools.count()
        self._load()

    def _load(self):
        self.mtimes = [p.stat().st_mtime for p in self.paths]
        urls = {}
        for p in self.paths:
            for k, v in json.loads(p.read_text()).items():
                urls.setdefault(k, []).extend(v)
        for k, v in getattr(self, "urls", {}).items():    # a service with no healthy replica in the new file keeps
            urls.setdefault(k, v)                          # its old ones: calls retry until one is back
        self.urls = urls
        self._checked = time.monotonic()

    def next(self, service: str) -> str:
        if time.monotonic() - self._checked > self.RELOAD_S:
            self._checked = time.monotonic()
            try:
                if [p.stat().st_mtime for p in self.paths] != self.mtimes:
                    self._load()
            except (OSError, ValueError):     # a file being replaced: keep the current list
                pass
        urls = self.urls.get(service)
        if not urls:
            raise RuntimeError(f"no healthy replica of '{service}' in {self.path}")
        now = time.monotonic()
        up = [u for u in urls if self.cool.get(u, 0.0) <= now] or urls
        k = next(self._turn) % len(up)
        return min(up[k:] + up[:k], key=lambda u: self.inflight[u])

    def failed(self, url: str):
        self.cool[url] = time.monotonic() + self.COOL_S

    def count(self, service: str) -> int:
        return len(self.urls.get(service, []))


class Client:
    def __init__(self, endpoints: Endpoints, limits: dict | None = None, timeout_s: float = 1800):
        self.ep = endpoints
        lim = dict(DEFAULT_LIMITS, **(limits or {}))
        self.sem = {k: asyncio.Semaphore(v * max(1, endpoints.count(k))) for k, v in lim.items()}
        self.timeout_s = timeout_s
        self.session = None

    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                                             connector=aiohttp.TCPConnector(limit=0))
        return self

    async def __aexit__(self, *a):
        await self.session.close()

    async def _post(self, service, path, *, json_body=None, data=None, tries=5):
        last = None
        async with self.sem[service]:
            for k in range(tries):
                base = self.ep.next(service)
                self.ep.inflight[base] += 1
                try:
                    async with self.session.post(base + path, json=json_body,
                                                 data=data() if callable(data) else data) as r:
                        if r.status == 200:
                            # vLLM's transcription route answers an exception it did not expect with HTTP 200
                            # and an {"error": ...} body, so a 200 is not yet an answer
                            out = await r.json()
                            err = out.get("error") if isinstance(out, dict) else None
                            if err is None:
                                return out
                            if empty_decode(err):
                                return {"text": "", "segments": []}
                            last = f"200 with an error: {str(err)[:300]}"
                        else:
                            body = (await r.text())[:300]
                            if r.status < 500:
                                raise ValueError(f"{service}{path} {r.status}: {body}")
                            last = f"{r.status}: {body}"
                        self.ep.failed(base)
                except (ValueError, asyncio.CancelledError):
                    raise
                except Exception as e:  # noqa: BLE001 - connection errors are retried on another replica
                    last = f"{type(e).__name__}: {e}"
                    self.ep.failed(base)
                finally:
                    self.ep.inflight[base] -= 1
                if k + 1 < tries:
                    await asyncio.sleep(min(30, 1.5 * 2 ** k))
        raise RuntimeError(f"{service}{path} failed after {tries} tries: {last}")

    # ------------------------------------------------------------------ Stage A: front server
    async def sed(self, x32k: np.ndarray) -> dict:
        r = await self._post("front", "/sed", json_body={"pcm16": enc_pcm16(x32k), "sr": 32000})
        return {"fps": r["fps"], **{g: dec_f16(v) for g, v in r["tracks"].items()}}

    async def separate(self, x44k: np.ndarray) -> np.ndarray:
        r = await self._post("front", "/separate", json_body={"pcm16": enc_pcm16(x44k), "sr": 44100})
        return np.stack([dec_f16(r[k]) for k in ("music", "sfx", "speech")])

    async def uvr(self, x44k: np.ndarray) -> np.ndarray:
        """Emilia-Pipe's separator (comparisons only): the vocals stem it keeps."""
        r = await self._post("uvr", "/uvr", json_body={"pcm16": enc_pcm16(x44k), "sr": 44100})
        return dec_f16(r["vocals"])

    # ------------------------------------------------------------------ Stage B: vLLM diarization
    async def diarize(self, path: Path, duration_s: float, tokens_per_s: float = 24.0, prompt: str = None,
                      tries: int = 5):
        """-> (turn text, budget hit). vLLM serves Who-Spoke-When for the `transcription` task only, so
        the call is /v1/audio/transcriptions, whose `prompt` field the model takes as its instruction
        (vllm moss_transcribe_diarize.get_generation_prompt) and whose audio it never cuts into clips
        (max_audio_clip_s=None). That route returns no token counts, so a pass that ran out of budget
        shows as text that stops before its closing `]`. Every call keeps the client's half-hour timeout, however
        long its audio: a long pass holds a large share of a server's KV cache while it decodes, so letting it run
        longer stalls every short call behind it (melati/diarize.py)."""
        import aiohttp
        from melati.diarize import PROMPT
        audio = Path(path).read_bytes()
        budget = int(min(tokens_per_s * duration_s + 128, 131072 - 14 * duration_s - 512))

        def form():
            f = aiohttp.FormData()
            f.add_field("file", audio, filename=Path(path).name, content_type="audio/flac")
            f.add_field("model", "melati-diar")
            f.add_field("prompt", prompt or PROMPT)
            f.add_field("temperature", "0")
            f.add_field("max_completion_tokens", str(budget))
            f.add_field("response_format", "json")
            return f
        r = await self._post("diar", "/v1/audio/transcriptions", data=form, tries=tries)
        text = (r.get("text") or "").strip()
        return text, bool(text) and not text.endswith("]")

    # ------------------------------------------------------------------ Stages D, H: vLLM Whisper
    async def transcribe(self, x16k: np.ndarray, language: str | None = None, temperature: float = 0.0,
                         max_tokens: int | None = None, seed: int | None = None) -> dict:
        import aiohttp
        buf = io.BytesIO()
        sf.write(buf, x16k, 16000, format="WAV", subtype="PCM_16")
        audio = buf.getvalue()

        def form():
            f = aiohttp.FormData()
            f.add_field("file", audio, filename="seg.wav", content_type="audio/wav")
            f.add_field("model", "melati-asr")
            f.add_field("response_format", "verbose_json")
            f.add_field("temperature", str(temperature))
            if language:
                f.add_field("language", language)
            if max_tokens:
                f.add_field("max_completion_tokens", str(max_tokens))
            if seed is not None:
                f.add_field("seed", str(seed))
            return f
        return await self._post("asr", "/v1/audio/transcriptions", data=form)

    # ------------------------------------------------------------------ Stage E: aligner
    async def align(self, items):
        """[(x16k, text, language)] -> [{"words", "score"} | {"error"}]"""
        r = await self._post("align", "/align", json_body={"items": [
            {"pcm16": enc_pcm16(x), "text": t, "language": l} for x, t, l in items]})
        return r["results"]

    # ------------------------------------------------------------------ Stage F: quality
    async def mos(self, x24k: np.ndarray) -> float:
        import aiohttp
        buf = io.BytesIO()
        sf.write(buf, x24k, 24000, format="WAV", subtype="PCM_16")
        audio = buf.getvalue()

        def form():
            f = aiohttp.FormData()
            f.add_field("file", audio, filename="seg.wav", content_type="audio/wav")
            return f
        return float((await self._post("mos", "/predict", data=form))["mos"])

    async def dnsmos(self, clips16k):
        r = await self._post("tag", "/dnsmos", json_body={"items": [{"pcm16": enc_pcm16(x)} for x in clips16k]})
        return r["scores"]

    # ------------------------------------------------------------------ audio LLM (vLLM chat with audio)
    async def chat_audio(self, service: str, x16k: np.ndarray, prompt: str, max_tokens: int = 512,
                         model: str | None = None, top_logprobs: int = 0):
        """-> the reply text; with top_logprobs > 0, (text, {token: logprob} of the FIRST generated token),
        which turns a one-word answer from a closed list into a distribution over the list."""
        buf = io.BytesIO()
        sf.write(buf, x16k, 16000, format="WAV", subtype="PCM_16")
        b64 = base64.b64encode(buf.getvalue()).decode()
        body = {"model": model or f"melati-{service}", "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": prompt}]}]}
        if top_logprobs:
            body.update(logprobs=True, top_logprobs=top_logprobs)
        r = await self._post(service, "/v1/chat/completions", json_body=body)
        text = r["choices"][0]["message"]["content"] or ""
        if not top_logprobs:
            return text
        content = ((r["choices"][0].get("logprobs") or {}).get("content") or [{}])[0]
        return text, {t["token"]: t["logprob"] for t in content.get("top_logprobs") or []}

    # ------------------------------------------------------------------ Stage G: emotion
    async def emotion(self, model: str, clips16k):
        r = await self._post("tag", "/emotion", json_body={"model": model,
                                                            "items": [{"pcm16": enc_pcm16(x)} for x in clips16k]})
        return np.asarray(r["probs"], np.float32)

    async def frames(self, model: str, clips16k):
        """A frame-level model (melati.nv_frames): per clip, (frames, classes) probabilities -> (arrays, classes, fps)."""
        r = await self._post("tag", "/frames", json_body={"model": model,
                                                           "items": [{"pcm16": enc_pcm16(x)} for x in clips16k]})
        return [np.asarray(f, np.float32).reshape(-1, len(r["classes"])) for f in r["frames"]], r["classes"], r["fps"]

    # ------------------------------------------------------------------ Stage sounds
    async def sounds_window(self) -> float:
        """The clip length the sounds server's tagger reads, in seconds: 10.24 for an EAT model, 10 for the ensemble
        (a server from before it said reads 10.24). Asked once."""
        if getattr(self, "_sounds_window", None) is None:
            async with self.session.get(self.ep.next("sounds") + "/health") as r:
                self._sounds_window = float((await r.json()).get("window_s") or 10.24)
        return self._sounds_window

    async def sounds(self, clips16k):
        """Melati's AudioSet tagger (serve/sounds_server.py): per clip of at most sounds_window() seconds, a
        probability for each of AudioSet's 527 classes -> (array [clips, 527], labels, model as repo@revision)."""
        r = await self._post("sounds", "/sounds", json_body={"items": [{"pcm16": enc_pcm16(x)} for x in clips16k]})
        return np.asarray(r["probs"], np.float32), r["labels"], r["model"]

    async def sound_frames(self, clips16k, classes):
        """Where each of `classes` sounds in each clip (any length): every 160 ms, the mean of the two frame models
        serve/sounds_server.py runs -> ([array [steps, len(classes)] per clip], step seconds, model)."""
        r = await self._post("sounds", "/frames", json_body={"classes": list(classes),
                                                              "items": [{"pcm16": enc_pcm16(x)} for x in clips16k]})
        return [np.asarray(f, np.float32).reshape(-1, len(classes)) for f in r["frames"]], r["step_s"], r["model"]
