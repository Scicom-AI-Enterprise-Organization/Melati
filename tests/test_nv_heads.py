"""The non-verbal heads' answers are read exactly as Qwen3-Omni's, each at its own labeller's thresholds."""
import numpy as np

from melati import nv_gaps as NG


def test_head_labels_are_omni_labels():
    assert set(NG.HEAD_LABELS) - {"other"} <= set(NG.LABELS)          # other: untagged, never a tag
    assert "other" not in NG.TO_CLASS
    assert len(set(NG.HEAD_LABELS)) == len(NG.HEAD_LABELS)


def test_decide_reads_a_head_distribution():
    dist = dict.fromkeys(NG.HEAD_LABELS, 0.0)
    dist.update({"speech": 0.3, "laugh": 0.6, "breath": 0.1})
    assert NG.decide(dist, {"laugh": 0.5}) == "laugh"
    assert NG.decide(dist, {"laugh": 0.7}) is None
    dist.update({"throat clearing": 0.65, "laugh": 0.2})
    assert NG.decide(dist, {"throat_clear": 0.6}) == "throat_clear"


def test_thresholds_follow_the_labeller():
    nested = {"heads": {"laugh": 0.4, "breath": 0.7}, "omni": {"laugh": 0.05, "cough": 0.95}}
    assert NG.thresholds_for({"by": "heads"}, nested) == {"laugh": 0.4, "breath": 0.7}
    assert NG.thresholds_for({"by": "omni"}, nested) == {"laugh": 0.05, "cough": 0.95}
    assert NG.thresholds_for({}, nested) == {"laugh": 0.05, "cough": 0.95}       # older tags.jsonl: Qwen3-Omni's
    flat = {"laugh": 0.05, "cough": 0.95}
    assert NG.thresholds_for({"by": "heads"}, flat) == flat


def test_a_breath_just_above_the_noise_floor_is_a_candidate():
    """The breaths the old rule missed sat in real gaps a dB or two under floor + 6 dB."""
    rng = np.random.default_rng(0)
    sr, n = NG.SR, int(3.0 * NG.SR)
    floor = 10 ** (-50 / 20)
    x = rng.normal(0, floor, n).astype(np.float32)                            # background at -50 dBFS
    words = [{"s": 0.3, "e": 0.8}, {"s": 1.6, "e": 2.1}]
    for w in words:                                                          # speech at about -20 dBFS
        a, b = int(w["s"] * sr), int(w["e"] * sr)
        x[a:b] += (10 ** (-20 / 20) * np.sin(2 * np.pi * 220 * np.arange(b - a) / sr) * np.sqrt(2)).astype(np.float32)
    a, b = int(1.0 * sr), int(1.3 * sr)                                       # a breath 4 dB over the floor
    x[a:b] = rng.normal(0, floor * 10 ** (4 / 20), b - a).astype(np.float32)
    old = NG.candidates(x, words, floor_db=6.0)
    new = NG.candidates(x, words)
    assert not any(c["slot"] == 1 for c in old)
    hit = [c for c in new if c["slot"] == 1]
    assert hit and 0.95 <= hit[0]["s"] <= 1.05 and 1.25 <= hit[0]["e"] <= 1.35, hit
