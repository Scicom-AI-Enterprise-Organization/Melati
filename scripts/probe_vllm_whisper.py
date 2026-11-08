#!/usr/bin/env python3
"""What does a vLLM Whisper server return when no language is given? One clip per language.

Stage D depends on the answer: it sends no language (code-switched speech) and keeps the one the
server reports. If the server cannot detect the language, it silently decodes everything as
English, which is the failure this probe exists to catch.

    python scripts/probe_vllm_whisper.py --server http://127.0.0.1:8199 \\
        --cv22 $MELATI_DATA/asr/cv22/test/manifest.csv --langs en zh id ja ta ar ru
"""
import argparse, csv, json, time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", default="melati-asr")
    ap.add_argument("--cv22", type=Path, required=True)
    ap.add_argument("--langs", nargs="+", default=["en", "zh", "id", "ja", "ta"])
    args = ap.parse_args()
    rows = list(csv.DictReader(args.cv22.open(encoding="utf-8")))
    for lang in args.langs:
        r = next((x for x in rows if x["lang"] == lang), None)
        if not r:
            print(lang, "no clip"); continue
        audio = (args.cv22.parent / r["audio_filepath"]).read_bytes()
        t = time.time()
        resp = requests.post(f"{args.server}/v1/audio/transcriptions",
                             files={"file": ("a.flac", audio, "audio/flac")},
                             data={"model": args.model, "response_format": "verbose_json", "temperature": "0"})
        d = resp.json() if resp.status_code == 200 else {"error": resp.status_code, "body": resp.text[:200]}
        print(json.dumps({"lang": lang, "ref": r["text"][:60], "ms": round((time.time() - t) * 1000),
                          "language": d.get("language"), "text": (d.get("text") or "")[:80],
                          "keys": sorted(d.keys()), "seg_keys": sorted((d.get("segments") or [{}])[0].keys()),
                          "error": d.get("error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
