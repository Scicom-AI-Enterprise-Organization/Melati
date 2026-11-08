#!/usr/bin/env python3
"""Sample Common Voice 22 per language: the replay half of the Melati ASR mix. RUNS ON THE BOX.

The hallucination recipe (arXiv 2609.32560) trains on non-speech with empty targets and on
synthesised hallucination phrases, and every one of its fine-tunes lost multilingual accuracy:
FLEURS macro CER 0.305 for base large-v3 against 0.381 at best. Its corpus is 75 languages but
Malay-weighted, so a thousand steps pull the decoder towards Malay. Replay is the standard cure
for that kind of forgetting: real, human-validated speech in every language the model should
keep, mixed into the same batches.

Source: `malaysia-ai/common_voice_22_0`, a mirror of Common Voice 22 (CC0). Its metadata is
three parquet splits; its audio is 70 zips of MP3, 286 GB, not grouped by language. Downloading
them to take a few hundred clips per language would be absurd, so this reads the zips' central
directories over HTTP and then fetches only the selected members with byte-range requests.

  * Only locales that map to a Whisper language token are kept. A clip in a language Whisper
    has no token for would have to be labelled as some other language, which teaches the
    decoder to emit the wrong language id.
  * Only validated clips: up_votes >= 2 and down_votes == 0.
  * Speakers are spread: each language is filled round-robin over `client_id`, so 400 clips
    are not 400 sentences from the three people who recorded the most.
  * `train` feeds the mix; a disjoint `test` sample (from CV22's own test split) is an
    in-domain multilingual check. FLEURS, which never enters training, stays the
    out-of-domain guard.

    python asr/scripts/build_cv22.py --out $MELATI_DATA/asr/audio_train/cv22 \\
        --per-lang 400 --test-per-lang 20 --workers 48
"""
import argparse, csv, io, json, os, random, struct, threading, time, zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

REPO = "malaysia-ai/common_voice_22_0"
SR = 16000

# Common Voice locale -> Whisper language code. Locales absent here have no Whisper token.
CV_TO_WHISPER = {
    "af": "af", "am": "am", "ar": "ar", "as": "as", "az": "az", "ba": "ba", "be": "be",
    "bg": "bg", "bn": "bn", "br": "br", "bs": "bs", "ca": "ca", "cs": "cs", "cy": "cy",
    "da": "da", "de": "de", "el": "el", "en": "en", "es": "es", "et": "et", "eu": "eu",
    "fa": "fa", "fi": "fi", "fo": "fo", "fr": "fr", "gl": "gl", "gu": "gu", "ha": "ha",
    "he": "he", "hi": "hi", "hr": "hr", "ht": "ht", "hu": "hu", "hy-AM": "hy", "id": "id",
    "is": "is", "it": "it", "ja": "ja", "ka": "ka", "kk": "kk", "km": "km", "kn": "kn",
    "ko": "ko", "lb": "lb", "ln": "ln", "lo": "lo", "lt": "lt", "lv": "lv", "mg": "mg",
    "mk": "mk", "ml": "ml", "mn": "mn", "mr": "mr", "ms": "ms", "mt": "mt", "my": "my",
    "nb-NO": "no", "ne-NP": "ne", "nl": "nl", "nn-NO": "nn", "oc": "oc", "pa-IN": "pa", "pl": "pl",
    "ps": "ps", "pt": "pt", "ro": "ro", "ru": "ru", "sa": "sa", "sd": "sd", "si": "si",
    "sk": "sk", "sl": "sl", "sn": "sn", "so": "so", "sq": "sq", "sr": "sr", "sv-SE": "sv",
    "sw": "sw", "ta": "ta", "te": "te", "tg": "tg", "th": "th", "tk": "tk", "tl": "tl",
    "tr": "tr", "tt": "tt", "uk": "uk", "ur": "ur", "uz": "uz", "vi": "vi", "yi": "yi",
    "yo": "yo", "yue": "yue", "zh-CN": "zh", "zh-TW": "zh", "zh-HK": "yue",
}


class Remote:
    """Byte ranges of one file on the Hub. The resolve URL redirects to a signed CDN URL that
    honours Range; the redirect target is cached because it is the same for every member."""

    def __init__(self, filename, token=None):
        import requests
        self.url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{filename}"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.s = requests.Session()
        self.lock = threading.Lock()
        self.target = None
        self.size = None

    def _resolve(self):
        with self.lock:
            if self.target is None:
                r = self.s.head(self.url, headers=self.headers, allow_redirects=True, timeout=60)
                r.raise_for_status()
                self.target = r.url
                self.size = int(r.headers["Content-Length"])
        return self.target

    def get(self, start, end, tries=5):
        """Bytes [start, end), retried: a sweep of 50k ranged GETs will meet a 5xx."""
        url = self._resolve()
        for k in range(tries):
            try:
                r = self.s.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
                if r.status_code in (403, 410):          # signed URL expired: resolve again
                    with self.lock:
                        self.target = None
                    url = self._resolve()
                    continue
                r.raise_for_status()
                if len(r.content) != end - start:
                    raise IOError(f"short read {len(r.content)} != {end - start}")
                return r.content
            except Exception:
                if k == tries - 1:
                    raise
                time.sleep(1.5 * (k + 1))


def central_directory(remote: Remote):
    """name -> (local header offset, compressed size, method) from the end of the zip."""
    remote._resolve()
    size = remote.size
    tail = remote.get(max(0, size - 65536 - 22), size)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise IOError("no end-of-central-directory record")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    n_entries = struct.unpack("<H", tail[i + 10:i + 12])[0]
    if cd_off == 0xFFFFFFFF or n_entries == 0xFFFF:    # zip64
        j = tail.rfind(b"PK\x06\x07")
        z64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        rec = remote.get(z64_off, z64_off + 56)
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    cd = remote.get(cd_off, cd_off + cd_size)
    out, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        hoff = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, hoff):          # zip64 extra field
            q, vals = 0, []
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals = list(struct.unpack(f"<{hsz // 8}Q", extra[q + 4:q + 4 + hsz]))
                    break
                q += 4 + hsz
            if usize == 0xFFFFFFFF:
                usize = vals.pop(0)
            if csize == 0xFFFFFFFF:
                csize = vals.pop(0)
            if hoff == 0xFFFFFFFF:
                hoff = vals.pop(0)
        out[name] = (hoff, csize, method)
        p += 46 + nlen + xlen + clen
    return out


def fetch_member(remote: Remote, hoff, csize, method):
    head = remote.get(hoff, hoff + 30)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    start = hoff + 30 + nlen + xlen
    data = remote.get(start, start + csize)
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise IOError(f"unsupported zip method {method}")


def to16k(raw: bytes):
    import librosa
    x, sr = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
    return x.astype("float32")


def pick(rows, n, rng):
    """Up to n rows, round-robin over speakers so no few voices dominate a language."""
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["client_id"]].append(r)
    spk = list(by_spk)
    rng.shuffle(spk)
    for s in spk:
        rng.shuffle(by_spk[s])
    out, k = [], 0
    while len(out) < n and spk:
        nxt = []
        for s in spk:
            if k < len(by_spk[s]):
                out.append(by_spk[s][k])
                if len(out) == n:
                    break
                nxt.append(s)
        spk, k = nxt, k + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-lang", type=int, default=400, help="train clips per language")
    ap.add_argument("--test-per-lang", type=int, default=20, help="test clips per language")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"],
                    help="build only these (a larger test sample is built on its own, into its own --out)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    tok = os.environ.get("HF_TOKEN")
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    files = [f.rfilename for f in HfApi(token=tok).dataset_info(REPO).siblings]
    split_files = {sp: sorted(f for f in files if f.startswith(f"data/{sp}-")) for sp in ("train", "test")}
    zips = sorted(f for f in files if f.startswith("audio-") and f.endswith(".zip"))

    picked = {}
    for sp, per in (("train", args.per_lang), ("test", args.test_per_lang)):
        if sp not in args.splits:
            picked[sp] = []
            continue
        by_lang = defaultdict(list)
        for f in split_files[sp]:
            t = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset", token=tok),
                              columns=["client_id", "path", "sentence", "up_votes", "down_votes", "locale"])
            for r in t.to_pylist():
                lang = CV_TO_WHISPER.get(r["locale"])
                if not lang or not (r["sentence"] or "").strip():
                    continue
                if (r["up_votes"] or 0) < 2 or (r["down_votes"] or 0) > 0:
                    continue
                r["lang"] = lang
                by_lang[lang].append(r)
        sel = []
        for lang in sorted(by_lang):
            # over-draw by half: a clip outside the duration window is only known after decoding
            sel += pick(by_lang[lang], int(per * 1.5) + 2, rng)
        picked[sp] = sel
        print(f"[{sp}] {len(sel)} candidates over {len({r['lang'] for r in sel})} languages", flush=True)

    # where each wanted member lives
    want = {r["path"] for sp in picked for r in picked[sp]}
    where = {}
    remotes = {z: Remote(z, tok) for z in zips}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(central_directory, remotes[z]): z for z in zips}
        for fut in as_completed(futs):
            z = futs[fut]
            for name, ent in fut.result().items():
                if name in want:
                    where[name] = (z, *ent)
    print(f"[zips] located {len(where)}/{len(want)} wanted members in {len(zips)} zips", flush=True)

    for sp in [s_ for s_ in ("test", "train") if s_ in args.splits]:   # test first: the benchmark waits on it
        flac = args.out / sp / "flac"
        flac.mkdir(parents=True, exist_ok=True)
        man = args.out / sp / "manifest.csv"
        have = {}
        if man.exists():
            for r in csv.DictReader(man.open(encoding="utf-8")):
                have[r["cv_path"]] = r
        need = defaultdict(int)
        for r in have.values():
            need[r["lang"]] += 1
        per = args.per_lang if sp == "train" else args.test_per_lang
        todo = [r for r in picked[sp] if r["path"] in where and r["path"] not in have]
        rows, lock = list(have.values()), threading.Lock()

        def work(r):
            with lock:
                if need[r["lang"]] >= per:
                    return None
            z, hoff, csize, method = where[r["path"]]
            try:
                x = to16k(fetch_member(remotes[z], hoff, csize, method))
            except Exception as e:
                return ("err", r["path"], f"{type(e).__name__}: {str(e)[:80]}")
            dur = len(x) / SR
            if not (args.min_dur <= dur <= args.max_dur):
                return ("dur", r["path"], dur)
            with lock:
                if need[r["lang"]] >= per:
                    return None
                need[r["lang"]] += 1
            stem = Path(r["path"]).stem
            sf.write(flac / f"{stem}.flac", x, SR, format="FLAC", subtype="PCM_16")
            return ("ok", {"audio_filepath": f"flac/{stem}.flac", "text": r["sentence"].strip(),
                           "lang": r["lang"], "locale": r["locale"], "client_id": r["client_id"],
                           "cv_path": r["path"], "duration_s": round(dur, 3),
                           "license": "CC0-1.0", "source": "Common Voice 22"})

        t0, n_ok, n_err, n_dur = time.time(), 0, 0, 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, r) for r in todo]), 1):
                res = fut.result()
                if res is None:
                    continue
                if res[0] == "ok":
                    rows.append(res[1]); n_ok += 1
                elif res[0] == "err":
                    n_err += 1
                else:
                    n_dur += 1
                if i % 2000 == 0:
                    print(f"   [{sp}] {i}/{len(todo)} ok={n_ok} err={n_err} out-of-range={n_dur} "
                          f"{i / (time.time() - t0):.1f}/s", flush=True)
        with man.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["audio_filepath", "text", "lang", "locale", "client_id",
                                              "cv_path", "duration_s", "license", "source"])
            w.writeheader(); w.writerows(rows)
        langs = defaultdict(int)
        for r in rows:
            langs[r["lang"]] += 1
        hours = sum(float(r["duration_s"]) for r in rows) / 3600
        print(f"[{sp}] {len(rows)} clips, {hours:.1f} h, {len(langs)} languages "
              f"(min {min(langs.values()) if langs else 0}/lang) -> {man}", flush=True)
        (args.out / sp / "languages.json").write_text(json.dumps(dict(sorted(langs.items())), indent=1))


if __name__ == "__main__":
    main()
