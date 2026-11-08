#!/usr/bin/env python3
"""What a WER is made of: substitutions, deletions, insertions, and which words are inserted. RUNS ON THE BOX.

A fine-tune that writes fillers ("erm", "uh") the reference leaves out pays for them as insertions, which is
a different thing from transcribing worse. Per run and arm: the counts, and the commonest inserted and deleted
words.

    python asr/scripts/error_kinds.py $MELATI_DATA/asr/runs/base_large_v3 $MELATI_DATA/asr/runs2/<run> --arm genuine
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from metrics import normalise  # noqa: E402


def align(ref, hyp):
    """Levenshtein alignment over words -> list of (op, ref_word, hyp_word)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None)); i -= 1
        else:
            ops.append(("I", None, hyp[j - 1])); j -= 1
    return ops[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--arm", default="genuine")
    args = ap.parse_args()
    for run in args.runs:
        f = next(iter(sorted((run / "bench").glob(f"*/{args.arm}.jsonl"))), None)
        if not f:
            continue
        c, ins, dele = Counter(), Counter(), Counter()
        nref = 0
        for r in map(json.loads, f.open(encoding="utf-8")):
            ref, hyp = normalise(r.get("reference_text") or "").split(), normalise(r["hyp"]).split()
            if not ref:
                continue
            nref += len(ref)
            for op, a, b in align(ref, hyp):
                c[op] += 1
                if op == "I":
                    ins[b] += 1
                elif op == "D":
                    dele[a] += 1
        print(f"{run.name[:56]:56s} WER {(c['S'] + c['D'] + c['I']) / nref:.4f} = S {c['S'] / nref:.4f} + "
              f"D {c['D'] / nref:.4f} + I {c['I'] / nref:.4f}  (ref words {nref})")
        print(f"   inserted: {ins.most_common(12)}")
        print(f"   deleted:  {dele.most_common(12)}")


if __name__ == "__main__":
    main()
