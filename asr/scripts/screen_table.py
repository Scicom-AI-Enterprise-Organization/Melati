#!/usr/bin/env python3
"""The merge screen in one table: the arms box_asr_merge_screen.sh scores, for any run directories. RUNS ON THE BOX.

    python asr/scripts/screen_table.py $MELATI_DATA/asr/runs/base_large_v3 $MELATI_DATA/asr/merges/*
"""
import glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lid_report import score  # noqa: E402
from metrics import normalise, overgeneration_ratio, wer  # noqa: E402


def arm(d, a):
    f = sorted(glob.glob(f"{d}/bench/*/{a}.jsonl"))
    return [json.loads(l) for l in open(f[0])] if f else []


def cwer(rs):
    rs = [r for r in rs if normalise(r.get("reference_text") or "")]
    nw = [len(normalise(r["reference_text"]).split()) for r in rs]
    return sum(wer(r["reference_text"], r["hyp"]) * n for r, n in zip(rs, nw)) / max(1, sum(nw))


def main():
    print(f"{'run':40s} {'sil':>5s} {'mus':>5s} {'nsp':>5s} {'loop':>5s} {'LS':>6s} {'MY':>6s} {'FLdet':>6s} {'FLemp':>6s} {'LID':>6s}")
    for d in sys.argv[1:]:
        fl = sorted(glob.glob(f"{d}/bench/*/fleurs.jsonl"))
        if not fl or not arm(d, "genuine"):
            print(f"{Path(d).name[:40]:40s} pending"); continue
        v = [sum(bool(normalise(r["hyp"])) for r in arm(d, a)) / max(1, len(arm(d, a))) for a in ("silence", "music", "nonspeech")]
        rd = arm(d, "reduplication")
        loop = sum(overgeneration_ratio(r["hyp"], r["meta"].get("unit", ""), int(r["meta"].get("n_repeats", 0) or 0)) > 1.5
                   for r in rd) / max(1, len(rd))
        s = score(Path(fl[0]))
        fe = sum(not normalise(r["hyp"]) for r in arm(d, "fleurs")) / max(1, len(arm(d, "fleurs")))
        print(f"{Path(d).name[:40]:40s} " + " ".join(f"{x:5.3f}" for x in v) + f" {loop:5.3f} {cwer(arm(d, 'librispeech_test_clean')):6.4f}"
              f" {cwer(arm(d, 'genuine')):6.4f} {s['macro_cer']:6.3f} {fe:6.4f} {s.get('lid_accuracy', float('nan')):6.3f}")


if __name__ == "__main__":
    main()
