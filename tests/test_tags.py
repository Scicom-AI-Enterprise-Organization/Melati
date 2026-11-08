from melati.tags import Annotation, Event, Span, Word, event_slot, is_filler, render, well_formed


def words(*spec):
    return [Word(t, s, e, filler=is_filler(t)) for t, s, e in spec]


def test_the_example_from_the_brief():
    w = words(("hello", 0.0, 0.4), ("sorry,", 1.2, 1.6), ("um,", 1.7, 1.9), ("I", 2.0, 2.1),
              ("just", 2.1, 2.3), ("got", 2.3, 2.5), ("the", 2.5, 2.6), ("results", 2.6, 3.1))
    ann = Annotation(words=w,
                     events=[Event("coughs", 0.6, 1.0), Event("laughs", 3.3, 3.9)],
                     emotion=[Span("happy", 0, 7)],
                     delivery=[Span("fast", 3, 7)])
    out = render(ann)
    assert out == "<happy>hello [coughs] sorry, um, <fast>I just got the results</fast></happy> [laughs]"
    assert well_formed(out)


def test_neutral_and_unknown_labels_are_untagged():
    w = words(("a", 0, 0.2), ("b", 0.3, 0.5))
    assert render(Annotation(words=w, emotion=[Span("neutral", 0, 1)])) == "a b"


def test_delivery_is_split_at_an_emotion_boundary():
    w = words(*[(t, i * 0.3, i * 0.3 + 0.2) for i, t in enumerate("one two three four".split())])
    ann = Annotation(words=w, emotion=[Span("sad", 0, 1), Span("angry", 2, 3)],
                     delivery=[Span("slow", 1, 2)])
    out = render(ann)
    assert out == "<sad>one <slow>two</slow></sad> <angry><slow>three</slow> four</angry>"
    assert well_formed(out)


def test_overlapping_emotion_spans_do_not_nest():
    w = words(*[(t, i * 0.3, i * 0.3 + 0.2) for i, t in enumerate("a b c d".split())])
    out = render(Annotation(words=w, emotion=[Span("happy", 0, 2), Span("sad", 1, 3)]))
    assert out == "<happy>a</happy> <sad>b c d</sad>"
    assert well_formed(out)


def test_event_goes_to_the_nearest_boundary():
    w = words(("a", 0.0, 0.5), ("b", 2.0, 2.5))
    assert event_slot(Event("sighs", 0.1, 0.4), w) == 0          # onset nearer the start of a
    assert event_slot(Event("sighs", 1.1, 1.6), w) == 1          # in the gap
    assert event_slot(Event("sighs", 2.8, 3.0), w) == 2          # after the last word


def test_filler_renderings():
    w = words(("so", 0, 0.2), ("erm", 0.3, 0.6), ("yes", 0.8, 1.0))
    a = Annotation(words=w)
    assert render(a) == "so erm yes"
    assert render(a, fillers_as="tag") == "so [hesitates] yes"
    assert render(a, fillers_as="drop") == "so yes"


def test_well_formed_rejects_bad_text():
    assert not well_formed("<happy>a")
    assert not well_formed("<happy>a <fast>b</happy></fast>")
    assert not well_formed("a [laughs b")
    assert well_formed("<happy>a [laughs] <fast>b</fast></happy>")


def test_chinese_is_written_without_spaces():
    w = words(("我", 0.0, 0.2), ("用", 0.2, 0.4), ("iPhone", 0.4, 0.9), ("拍", 0.9, 1.1), ("的", 1.1, 1.2),
              ("，", 1.2, 1.2), ("好", 1.3, 1.5), ("看", 1.5, 1.7))
    assert render(Annotation(words=w)) == "我用 iPhone 拍的，好看"
    w = words(("你", 0.0, 0.2), ("好", 0.2, 0.4), ("吗", 0.5, 0.7))
    assert render(Annotation(words=w, events=[Event("laughs", 0.42, 0.48)])) == "你好 [laughs] 吗"
