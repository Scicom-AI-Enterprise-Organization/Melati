from melati.stage_seg import merge_same_speaker


def test_nested_turns_of_one_speaker_merge():
    # observed in a podcast: an S01 turn inside another S01 turn, with S02 overlapping both
    turns = [(1151.0, 1154.69, "S01"), (1152.41, 1153.88, "S01"), (1151.15, 1151.66, "S02"),
             (1154.57, 1155.69, "S01")]
    assert merge_same_speaker(turns) == [(1151.0, 1155.69, "S01"), (1151.15, 1151.66, "S02")]


def test_separate_turns_stay_separate():
    turns = [(0.0, 1.0, "A"), (1.5, 2.0, "A"), (0.5, 1.2, "B")]
    assert merge_same_speaker(turns) == turns[:1] + [(0.5, 1.2, "B"), (1.5, 2.0, "A")]


import numpy as np  # noqa: E402

from melati.stage_seg import SR, back_off, segment, subtract  # noqa: E402

HOP = 512 / SR


def probs_with_dips(dur, dips, speech=0.9, dip=0.1):
    """Silero-like frame probabilities: speech everywhere, a near-pause frame at each time in `dips`."""
    p = np.full(int(dur / HOP) + 1, speech, np.float32)
    for t in dips:
        p[int(t / HOP)] = dip
    return p


def test_subtract_flags_the_sides_a_cut_made():
    assert subtract([(0.0, 10.0)], [(6.0, 8.0)]) == [(0.0, 6.0, False, True), (8.0, 10.0, True, False)]


def test_a_cut_at_the_islands_own_edge_is_not_a_cut():
    # the overlap starts 0.1 s before the island would have ended anyway: jitter, not a word cut in two
    assert subtract([(0.0, 6.1)], [(6.0, 8.0)], islands=[(0.0, 6.1)]) == [(0.0, 6.0, False, False)]


def test_a_cut_end_backs_off_to_the_last_near_pause():
    p = probs_with_dips(12, [2.0, 5.0])        # the dip at 2.0 is beyond the 1.5 s window
    s, e, cs, ce = back_off((0.0, 6.0, False, True), p)
    assert s == 0.0 and abs(e - 5.0) < 2 * HOP and ce


def test_a_cut_start_moves_forward_to_the_first_near_pause():
    p = probs_with_dips(12, [8.5, 9.5])
    s, e, cs, ce = back_off((8.0, 10.0, True, False), p)
    assert abs(s - 8.5) < 2 * HOP and e == 10.0 and cs


def test_a_cut_side_with_no_dip_moves_to_the_lowest_frame_nearby():
    p = probs_with_dips(12, [])
    p[int(5.2 / HOP)] = 0.6                       # not a pause, but the least speech-like frame within 1.5 s
    s, e, cs, ce = back_off((0.0, 6.0, False, True), p)
    assert abs(e - 5.2) < 2 * HOP


def test_a_piece_too_short_to_back_off_is_dropped():
    assert back_off((0.0, 0.25, False, True), probs_with_dips(12, [])) is None


def test_an_uncut_piece_is_left_alone():
    piece = (1.0, 5.0, False, False)
    assert back_off(piece, probs_with_dips(12, [])) is piece


def test_overlap_ends_a_segment_at_a_pause_with_no_pad_into_the_overlap():
    # A talks 0-10 without a pause long enough for the VAD; B starts talking over A at 6
    x = np.zeros(int(14 * SR), np.float32)
    turns = [(0.0, 10.0, "A"), (6.0, 14.0, "B")]
    islands = [(0.0, 10.0), (10.0, 14.0)]
    p = probs_with_dips(14, [2.0, 5.0, 11.0])
    segs, _ = segment("r", x, turns, islands=islands, probs=p)
    a = [r for r in segs if r["speaker"] == "A"][0]
    b = [r for r in segs if r["speaker"] == "B"][0]
    assert a["end_backed_off"] and abs(a["end"] - 5.0) < 2 * HOP          # at the dip, nothing past it
    assert a["end"] <= 6.0 - 0.1 and b["start"] >= 10.0 + 0.1               # neither reaches into the overlap
