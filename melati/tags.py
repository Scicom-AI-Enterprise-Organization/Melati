"""Time-stamped events and their rendering as Drama-3 style tagged text.

The events are the source of truth; the tagged string is one rendering of them, so a new target
format never needs re-annotation. Syntax follows Fish Audio Drama 3 (arXiv 2610.18010, Appendix
C, and the `studio_content.json` of its Space):

    point tag    [laughs]               acts at a position (a word boundary)
    scoped tag   <happy> ... </happy>   acts on the words it encloses

    <happy>hello [coughs] sorry, um, <fast>I just got the results</fast></happy> [laughs]

Rules this module enforces, so every rendering is well formed:
  * emotion scopes partition a segment's words into runs; they never nest in each other, and
    neutral is left untagged.
  * delivery scopes (fast, slow, whisper, shout) sit inside the emotion scope they start in, and
    are split at an emotion boundary rather than crossing it.
  * point tags sit at the word boundary nearest the event's onset. An event that overlaps a
    word (laughing while speaking) still gets its tag at a boundary; its overlap is kept on the
    event, not invented in the text.
  * fillers are words. They are spoken, so they are written ("um", "erm", "emm"), and the
    aligner times them like any other word.
  * AudioSet sounds ([music], [applause], [dog]) are point tags too, placed the same way. Their names are
    AudioSet's (melati/sounds_inline.json), so an event from the sounds stage carries source "sounds"
    rather than a kind from NONVERBAL.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# The closed sets. Point-tag names follow Drama-3's verb forms; emotions are emotion2vec+'s
# classes without neutral/other/unknown.
NONVERBAL = ("laughs", "chuckles", "sighs", "gasps", "coughs", "sneezes", "sniffs", "clears throat",
             "breathes", "cries", "yawns", "hums", "screams", "groans")
EMOTIONS = ("happy", "sad", "angry", "surprised", "fearful", "disgusted")
DELIVERY = ("fast", "slow", "whisper", "shout")
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "err", "ah", "ahh", "eh", "ehh", "hmm", "hm", "mm",
           "mmm", "emm", "em", "aaa", "aa", "eee", "ee", "uhm", "ehm", "huh",
           "嗯", "呃", "啊", "额", "那个"}
# Chinese and Japanese are written without spaces. The aligner times each character as a word, so joining words
# with spaces wrote "你 好 吗"; a space between two of these characters is an artefact, never a pause.
_CJK = "\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef"
_CJK_GAP = re.compile(f"(?<=[{_CJK}]) (?=[{_CJK}])")


@dataclass
class Word:
    text: str
    start: float
    end: float
    conf: float = 1.0
    filler: bool = False


@dataclass
class Event:
    """A non-verbal sound: `kind` from NONVERBAL, times in seconds from the segment start."""
    kind: str
    start: float
    end: float
    conf: float = 1.0
    source: str = ""
    overlap: float = 0.0        # fraction of the event that falls inside a word


@dataclass
class Span:
    """An emotion or delivery label over words [first, last] (inclusive word indices)."""
    label: str
    first: int
    last: int
    conf: float = 1.0
    source: str = ""


@dataclass
class Annotation:
    words: list[Word] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    emotion: list[Span] = field(default_factory=list)
    delivery: list[Span] = field(default_factory=list)

    def to_json(self) -> dict:
        return {k: [asdict(x) for x in getattr(self, k)] for k in ("words", "events", "emotion", "delivery")}

    @classmethod
    def from_json(cls, d: dict) -> "Annotation":
        return cls(words=[Word(**w) for w in d.get("words", [])],
                   events=[Event(**e) for e in d.get("events", [])],
                   emotion=[Span(**s) for s in d.get("emotion", [])],
                   delivery=[Span(**s) for s in d.get("delivery", [])])


def is_filler(token: str) -> bool:
    t = token.strip().strip(".,!?;:\"'").lower()
    return t in FILLERS


def event_slot(ev: Event, words: list[Word]) -> int:
    """Index of the word boundary an event belongs at: 0 = before the first word,
    len(words) = after the last. The nearest boundary to the onset, where a boundary between
    words i-1 and i sits at the middle of the gap between them."""
    if not words:
        return 0
    bounds = [words[0].start] + [(words[i - 1].end + words[i].start) / 2 for i in range(1, len(words))] \
        + [words[-1].end]
    return min(range(len(bounds)), key=lambda i: abs(bounds[i] - ev.start))


def overlap_fraction(ev: Event, words: list[Word]) -> float:
    dur = max(ev.end - ev.start, 1e-6)
    inside = sum(max(0.0, min(ev.end, w.end) - max(ev.start, w.start)) for w in words)
    return min(1.0, inside / dur)


def _normalise_spans(spans: list[Span], n: int) -> list[Span]:
    """Clip to [0, n-1], drop empty ones, sort, and resolve overlaps (the later start wins)."""
    out = []
    for s in sorted(spans, key=lambda s: (s.first, s.last)):
        a, b = max(0, s.first), min(n - 1, s.last)
        if a > b:
            continue
        if out and a <= out[-1].last:
            out[-1] = Span(out[-1].label, out[-1].first, a - 1, out[-1].conf, out[-1].source)
            if out[-1].first > out[-1].last:
                out.pop()
        out.append(Span(s.label, a, b, s.conf, s.source))
    return out


def render(ann: Annotation, emotions: bool = True, delivery: bool = True, events: bool = True,
           fillers_as: str = "words", sounds: bool = True) -> str:
    """Drama-3 style text. `fillers_as`: "words" writes them as spoken, "tag" writes
    [hesitates] in their place, "drop" removes them (what a plain ASR transcript would say).
    `sounds` False leaves out the AudioSet sound tags and keeps the non-verbal ones."""
    words = ann.words
    n = len(words)
    emo = _normalise_spans([s for s in ann.emotion if s.label in EMOTIONS], n) if emotions else []
    dlv = _normalise_spans([s for s in ann.delivery if s.label in DELIVERY], n) if delivery else []

    # delivery spans are split wherever they would cross an emotion boundary
    if emo and dlv:
        cuts = sorted({s.first for s in emo} | {s.last + 1 for s in emo})
        split = []
        for d in dlv:
            a = d.first
            for c in cuts:
                if a < c <= d.last:
                    split.append(Span(d.label, a, c - 1, d.conf, d.source))
                    a = c
            split.append(Span(d.label, a, d.last, d.conf, d.source))
        dlv = [s for s in split if s.first <= s.last]

    opens = [[] for _ in range(n + 1)]
    closes = [[] for _ in range(n + 1)]
    for s in emo:                      # emotion opens first and closes last: it is the outer scope
        opens[s.first].append(s.label)
        closes[s.last + 1].insert(0, s.label)
    for s in dlv:
        opens[s.first].append(s.label)
        closes[s.last + 1].insert(0, s.label)
    # an inner (delivery) scope must close before the outer one does
    for i in range(n + 1):
        closes[i].sort(key=lambda lab: lab in EMOTIONS)

    points = [[] for _ in range(n + 1)]
    if events:
        for ev in sorted(ann.events, key=lambda e: e.start):
            if ev.kind in NONVERBAL or (sounds and ev.source == "sounds"):
                points[event_slot(ev, words)].append(f"[{ev.kind}]")

    out = []
    for i in range(n + 1):
        out += [f"</{lab}>" for lab in closes[i]]
        # a point tag at a boundary sits outside a scope that closes there and inside one that
        # opens there only when it is the first thing said in it; keep it simple and put tags
        # between the closes and the opens
        out += points[i]
        out += [f"<{lab}>" for lab in opens[i]]
        if i < n:
            w = words[i]
            if w.filler and fillers_as == "drop":
                continue
            out.append("[hesitates]" if (w.filler and fillers_as == "tag") else w.text)
    text = " ".join(out)
    # tidy: no space just inside a scope, none before punctuation
    for lab in set(EMOTIONS) | set(DELIVERY):
        text = text.replace(f"<{lab}> ", f"<{lab}>").replace(f" </{lab}>", f"</{lab}>")
    for p in (" ,", " .", " ?", " !", " ;", " :"):
        text = text.replace(p, p[1:])
    return _CJK_GAP.sub("", text).strip()


def well_formed(text: str) -> bool:
    """True if every scope opens and closes in order and no square tag is unclosed."""
    import re
    stack = []
    for m in re.finditer(r"<(/?)([a-z ]+)>|\[([^\]]*)\]|\[|\]", text):
        if m.group(0) in ("[", "]"):
            return False
        if m.group(2):
            if m.group(1):
                if not stack or stack.pop() != m.group(2):
                    return False
            else:
                stack.append(m.group(2))
    return not stack
