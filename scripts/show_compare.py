#!/usr/bin/env python3
"""Print scripts/compare_emilia.py's JSON as one table per measure, pipelines side by side.

    python scripts/show_compare.py results/compare/voxconverse.json
"""
import json, sys

ROWS = [
    ("yield", "segments", "segments"), ("yield", "hours", "hours kept"), ("yield", "share_of_input", "share of input"),
    ("yield", "duration_s", "duration p10/50/90/99 (s)"), ("yield", "share_lt_3s", "share < 3 s"),
    ("yield", "share_ge_29_5s", "share >= 29.5 s"),
    ("audio", "codecs", "codec"), ("audio", "bitrate_kbps", "bitrate p50 (kbit/s)"),
    ("audio", "cutoff_khz", "spectral cutoff p10/50/90 (kHz)"), ("audio", "lufs_iqr", "loudness IQR (LU)"),
    ("audio", "share_peak_at_full_scale", "peak at full scale"), ("audio", "share_over_full_scale", "over full scale"),
    ("audio", "decoded_minus_stated_ms", "decoded - stated length (ms)"),
    ("audio", "share_start_inside_sound", "edge energy near speech: start"),
    ("audio", "share_end_inside_sound", "edge energy near speech: end"),
    ("purity", "purity_weighted", "purity (duration-weighted)"), ("purity", "share_purity_lt_0_9", "purity < 0.9"),
    ("purity", "share_second_speaker_ge_0_5s", "second speaker >= 0.5 s"),
    ("purity", "share_overlap_ge_0_2s", "overlap >= 0.2 s"),
    ("purity", "share_mostly_no_reference_speech", "mostly no reference speech"),
    ("purity", "single_speaker_speech_kept", "single-speaker speech kept"),
    ("code_switch", "english_word_share", "English words in Malay segments"),
    ("code_switch", "share_code_switched", "Malay segments with >= 2 English words"),
]


def fmt(v):
    if isinstance(v, dict):
        if set(v) <= {"10", "50", "90", "99", "5", "95"}:
            return "/".join("-" if v[k] is None else f"{v[k]:g}" for k in sorted(v, key=float))
        return ", ".join(f"{k} {n}" for k, n in v.items())
    if isinstance(v, float):
        return f"{v:.4g}"
    return "-" if v is None else str(v)


def main():
    d = json.load(open(sys.argv[1]))
    names = list(d["pipelines"])
    print(f"{d['recordings']} recordings, {d['input_hours']} h; upstream left out: {d['not_processed_by_emilia']}")
    w = 34
    print(f"{'':{w}}" + "".join(f"{n:>26}" for n in names))
    for sec, key, label in ROWS:
        vals = [d["pipelines"][n].get(sec, {}).get(key) for n in names]
        if all(v is None for v in vals):
            continue
        if sec == "audio" and key == "bitrate_kbps":
            vals = [v and v.get("50") for v in vals]
        print(f"{label:{w}}" + "".join(f"{fmt(v)[:25]:>26}" for v in vals))
    ea = d.get("edges_by_alignment")
    if ea:
        print("\nwords at the cut (own transcript aligned to own audio):")
        for n, v in ea.items():
            print(f"  {n:22s} first {fmt(v['share_first_word_at_cut'])}  last {fmt(v['share_last_word_at_cut'])}  "
                  f"either {fmt(v['share_either'])}  ({v['aligned']} of {v['of']} aligned)")
    nv = d.get("nonverbal")
    if nv:
        print(f"\nnon-verbal: {nv['recordings']} recordings, source events (p >= {nv['floor']}): {nv['source_events']}")
        for name in [k for k in nv if isinstance(nv[k], dict) and k not in ("source_events",)]:
            print(f"  {name}: " + "; ".join(f"{c} {fmt(v)}" for c, v in nv[name].items()))
    print("\nthroughput:", json.dumps(d.get("throughput")))


if __name__ == "__main__":
    main()
