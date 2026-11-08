#!/usr/bin/env python3
"""Do the released speaker ids hold one voice each? WeSpeaker's ResNet34 (VoxCeleb, ONNX; scripts/fetch_wespeaker.py)
embeds clips of random recordings read back from the published shards. In recordings with several ids, the cosine
between two clips of one id and of two ids sets the threshold where the two errors are equal; in recordings released
with one id, the clips are then grouped at that threshold, and one that falls into two groups of real size holds a
second voice under a single id. RUNS ON THE BOX.

    HF_HOME=$MELATI_DATA/hf python scripts/release_speakers.py --hf $MELATI_DATA/podcast/hf \\
        --out $MELATI_DATA/results/dataset/release_speakers.json
"""
import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_S, MAX_S = 2.0, 8.0


def embedder(path):
    import onnxruntime as ort
    import torch
    import torchaudio
    from scipy.signal import resample_poly
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 16
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    def embed(x24):
        x = resample_poly(x24[:int(MAX_S * 24000)], 2, 3).astype(np.float32)
        w = torch.from_numpy(x).unsqueeze(0) * (1 << 15)
        f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                                              sample_frequency=16000, window_type="hamming", energy_floor=0.0)
        f = (f - f.mean(0)).numpy()[None]
        e = sess.run(None, {"feats": f})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)
    return embed


def spread(rs, k):
    rs = sorted(rs, key=lambda r: r["start"])
    return rs if len(rs) <= k else [rs[int(i * len(rs) / k)] for i in range(k)]


def main():
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import AgglomerativeClustering
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--single", type=int, default=150)
    ap.add_argument("--multi", type=int, default=80)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    embed = embedder(hf_hub_download("Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx"))
    files = sorted((a.hf / "data").glob("train-*.parquet"))
    meta, where = defaultdict(list), {}
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=["id", "recording", "speaker", "start", "duration"]).to_pylist():
                meta[r["recording"]].append(r)
                where[r["id"]] = (f, g)
    long_ = {rec: [r for r in rs if r["duration"] >= MIN_S] for rec, rs in meta.items()}
    single = [rec for rec, rs in meta.items() if len({r["speaker"] for r in rs}) == 1 and len(long_[rec]) >= 6]
    multi = []
    for rec, rs in long_.items():
        by = defaultdict(list)
        for r in rs:
            by[r["speaker"]].append(r)
        if sum(len(v) >= 3 for v in by.values()) >= 2:
            multi.append(rec)
    rnd = random.Random(0)
    single, multi = rnd.sample(single, min(a.single, len(single))), rnd.sample(multi, min(a.multi, len(multi)))

    def clips(rows):
        need = defaultdict(set)
        for r in rows:
            need[where[r["id"]]].add(r["id"])
        out = {}
        for (f, g), ids in need.items():
            for r in pq.ParquetFile(f).read_row_group(g, columns=["id", "audio"]).to_pylist():
                if r["id"] in ids:
                    x, _ = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    out[r["id"]] = embed(x)
        return out

    same, diff = [], []
    for rec in multi:
        by = defaultdict(list)
        for r in long_[rec]:
            by[r["speaker"]].append(r)
        chosen = [r for v in by.values() if len(v) >= 3 for r in spread(v, 8)]
        e = clips(chosen)
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                c = float(e[chosen[i]["id"]] @ e[chosen[j]["id"]])
                (same if chosen[i]["speaker"] == chosen[j]["speaker"] else diff).append(c)
    same, diff = np.array(same), np.array(diff)
    ts = np.linspace(-0.2, 0.9, 221)
    gap = [abs((same < t).mean() - (diff >= t).mean()) for t in ts]
    t = float(ts[int(np.argmin(gap))])
    eer = float(((same < t).mean() + (diff >= t).mean()) / 2)
    print(f"{len(multi)} recordings with several ids: {len(same)} same-id pairs (cosine median {np.median(same):.3f}),"
          f" {len(diff)} two-id pairs (median {np.median(diff):.3f}); equal-error threshold {t:.3f}, EER {eer:.3f}")

    two_voices, report = [], []
    for rec in single:
        chosen = spread(long_[rec], 16)
        e = clips(chosen)
        X = np.stack([e[r["id"]] for r in chosen])
        lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=1 - t).fit(X).labels_
        sizes = sorted(np.bincount(lab), reverse=True)
        big = [s for s in sizes if s >= max(2, 0.25 * len(chosen))]
        sim = X @ X.T
        report.append({"recording": rec, "clips": len(chosen), "groups": [int(k) for k in sizes],
                       "min_cos": round(float(sim.min()), 3), "median_cos": round(float(np.median(sim[np.triu_indices(len(X), 1)])), 3)})
        if len(big) >= 2:
            two_voices.append(report[-1])
    print(f"{len(single)} recordings with one id: {len(two_voices)} fall into two or more groups of a quarter of their"
          f" clips or more ({100 * len(two_voices) / max(1, len(single)):.1f}%)")
    med = np.array([r["median_cos"] for r in report])
    print(f"  median cosine between a recording's clips: median {np.median(med):.3f}, 10th pct {np.percentile(med, 10):.3f}")

    # recordings diarized in one pass with several ids: a clip nearer another id's voice than its own is misassigned
    one_pass = [rec for rec in multi if not any("_w" in r["speaker"] for r in meta[rec])]
    wrong, total, per_rec = 0, 0, []
    for rec in one_pass:
        by = defaultdict(list)
        for r in long_[rec]:
            by[r["speaker"]].append(r)
        chosen = [r for v in by.values() if len(v) >= 3 for r in spread(v, 12)]
        e = clips(chosen)
        ids = sorted({r["speaker"] for r in chosen})
        X = {r["id"]: e[r["id"]] for r in chosen}
        w = 0
        for r in chosen:
            mine = [X[o["id"]] for o in chosen if o["speaker"] == r["speaker"] and o["id"] != r["id"]]
            own = float(np.mean(mine, 0) @ X[r["id"]])
            other = max(float(np.mean([X[o["id"]] for o in chosen if o["speaker"] == s_], 0) @ X[r["id"]])
                        for s_ in ids if s_ != r["speaker"])
            w += other > own
        wrong += w
        total += len(chosen)
        per_rec.append(w / len(chosen))
    print(f"{len(one_pass)} one-pass recordings with several ids: {wrong} of {total} clips sit nearer another id's"
          f" voice than their own ({100 * wrong / max(1, total):.1f}%); per recording {np.percentile(per_rec, [50, 90]).round(3).tolist()}")
    for r in two_voices[:15]:
        print(f"    {r['recording']} groups {r['groups']} median cos {r['median_cos']} [{meta[r['recording']][0]['id']}]")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"threshold": t, "eer": eer, "same_median": float(np.median(same)),
                                 "diff_median": float(np.median(diff)), "single_checked": len(single),
                                 "single_two_voices": len(two_voices), "single": report}, indent=1))


if __name__ == "__main__":
    main()
