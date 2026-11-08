#!/usr/bin/env python3
"""Metrics for scoring a Whisper hypothesis for hallucination and repetition.

Kept dependency-free so it can be vendored onto the GPU box without an install step.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

__all__ = [
    "normalise", "max_ngram_repeat", "ngram_dup_rate", "unique_word_share",
    "longest_char_run", "overgeneration_ratio", "wer", "cer", "load_lexicon", "in_lexicon",
]

_PUNCT = re.compile(r"[.?!,;:\"'`´()\[\]…]+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").strip().casefold()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def max_ngram_repeat(text: str, n: int = 1) -> int:
    """Longest run of the SAME n-gram repeating back-to-back.

    This is the loop signal. `ngram_dup_rate` counts duplicates anywhere, which a
    legitimately repetitive transcript also trips; consecutive runs do not.
    """
    words = normalise(text).split()
    if len(words) < n:
        return 0
    best = 1
    for i in range(len(words) - n + 1):
        block = words[i:i + n]
        run, j = 1, i + n
        # Step by n, not 1: walking overlapping positions splices the (a,b) and (b,a)
        # chains of "a b a b a b" into one run and reports 4 instead of 3.
        while words[j:j + n] == block:
            run, j = run + 1, j + n
        best = max(best, run)
        if best > (len(words) - i) // n:
            break
    return best


def ngram_dup_rate(text: str, n: int = 5) -> float:
    """Share of n-grams that are not unique. The Whisper paper's repetition proxy."""
    words = normalise(text).split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def unique_word_share(text: str) -> float:
    """NeMo SDP's `hall_repeated_ngrams` feature: <= 0.4 flags a loop."""
    words = normalise(text).split()
    return len(set(words)) / len(words) if words else 1.0


def longest_char_run(text: str) -> int:
    """Longest repeated character substring run, for sub-word loops like `tutututu`
    that never produce a space and so look like one long word."""
    s = normalise(text).replace(" ", "")
    if not s:
        return 0
    best = 0
    for size in range(1, min(8, len(s) // 2 + 1)):
        i = 0
        while i + size <= len(s):
            run, j = 1, i + size
            while j + size <= len(s) and s[j:j + size] == s[i:i + size]:
                run, j = run + 1, j + size
            best = max(best, run)
            i += 1
    return best


def overgeneration_ratio(hyp: str, unit: str, n_true: int) -> float:
    """How many times the unit appears in the hypothesis, over how many were spoken.

    1.0 is correct. >1 means the decoder kept going. The reduplication arm's whole point.
    """
    if n_true <= 0:
        return 0.0
    h, u = normalise(hyp).replace(" ", ""), normalise(unit).replace(" ", "")
    if not u:
        return 0.0
    return (h.count(u) if u else 0) / n_true


def _edit(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r = normalise(ref).split()
    return _edit(r, normalise(hyp).split()) / len(r) if r else 0.0


def cer(ref: str, hyp: str) -> float:
    r = list(normalise(ref).replace(" ", ""))
    return _edit(r, list(normalise(hyp).replace(" ", ""))) / len(r) if r else 0.0


def load_lexicon(path: Path, langs: set[str] | None = None) -> set[str]:
    """Load combined_lexicon.csv into a set of normalised phrases."""
    out = set()
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if langs and row["lang"] not in langs:
                continue
            if row["phrase_norm"]:
                out.add(row["phrase_norm"])
    return out


def in_lexicon(text: str, lexicon: set[str]) -> bool:
    """Whole hypothesis is a known hallucination phrase (NeMo's `hall_frequent_single_word`)."""
    return normalise(text) in lexicon
