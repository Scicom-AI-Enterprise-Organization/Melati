#!/usr/bin/env python3
"""The plain-PyTorch members (audioset/hub/melati_ensemble.py) against the released code they replace (the repo's
melati_ensemble.py at OLD, which runs each member's own code through transformers' trust_remote_code), on the same
clips of AudioSet's evaluation set: each member's probabilities in fp32 and in bf16. With --full, the new code alone
on the whole evaluation set: each member's mAP and the average's, fp32 and bf16, against the release check's
(results/audioset/ensemble_fp32.json). Writes results/audioset/ensemble_native.json. RUNS ON THE BOX, in .venv-sed.

    python audioset/check_native.py [--n 512] [--full]
"""
import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from data import load  # noqa: E402
from evaluate import mean_ap  # noqa: E402
from release import RESULTS  # noqa: E402

REPO = "Scicom-intl/Melati-AudioSet-Ensemble"
OLD = "98e419ca989e4162636caba4ca96606f52cb0b21"        # the release results/audioset/ensemble_fp32.json checked
OUT = RESULTS / "ensemble_native.json"


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(model, W, batch):
    """-> ({member: (clips, 527)}, seconds)."""
    per = {k: [] for k in model.members}
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, len(W), batch):
        _, each = model([np.asarray(w, np.float32) for w in W[i:i + batch]], batch=batch, each=True)
        for k, p in each.items():
            per[k].append(p)
    torch.cuda.synchronize()
    return {k: np.concatenate(v) for k, v in per.items()}, time.time() - t0


def free():
    gc.collect()
    torch.cuda.empty_cache()


def frontends(new, W):
    """The ported preprocessing against torchaudio's own, which the members were scored with: mel banks, Kaldi
    filterbanks (both windows), CED-base's extractor on a clip given alone, and the resampler -> max |difference|."""
    import torchaudio
    from torchaudio.compliance import kaldi
    from transformers import AutoFeatureExtractor
    banks, _ = kaldi.get_mel_banks(128, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    w = torch.from_numpy(np.stack([np.asarray(x, np.float32) for x in W[:8]]))
    out = {"kaldi_banks": float((new._kaldi_banks() - torch.nn.functional.pad(banks, (0, 1))).abs().max()),
           "ced_banks": float((new._ced_banks() - torchaudio.functional.melscale_fbanks(257, 0.0, 8000.0, 64, 16000))
                              .abs().max())}
    eat = torch.stack([kaldi.fbank((x - x.mean())[None], htk_compat=True, sample_frequency=16000, use_energy=False,
                                   window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10) for x in w])
    beats = torch.stack([kaldi.fbank(x[None] * 2 ** 15, num_mel_bins=128, sample_frequency=16000, frame_length=25,
                                     frame_shift=10) for x in w])
    out["fbank_eat"] = float((new.kaldi_fbank(w - w.mean(1, keepdim=True)) - eat).abs().max())
    out["fbank_beats"] = float((new.kaldi_fbank(w * 2 ** 15, "povey") - beats).abs().max())
    fe = AutoFeatureExtractor.from_pretrained("mispeech/ced-base", trust_remote_code=True,
                                              revision="db3e14a8db4c21b56b165261c39649741a900e7f")
    ced = torch.cat([fe(x.numpy(), sampling_rate=16000, return_tensors="pt")["input_values"] for x in w])
    out["ced_mel_db"] = float((new.ced_mel(w) - ced).abs().max())
    x = torch.from_numpy(np.asarray(W[0], np.float32))
    up = torchaudio.functional.resample(x, 16000, 44100)
    out["resample_up"] = float((new.resample(x, 16000, 44100) - up).abs().max())
    out["resample_down"] = float((new.resample(up, 44100) - torchaudio.functional.resample(up, 44100, 16000)).abs().max())
    print(f"[native] frontends, max |ours - torchaudio's|: {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=512, help="clips for the old-against-new comparison")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--full", action="store_true", help="also score the new code on the whole evaluation set")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(REPO, revision=OLD))
    old = module("old_ensemble", path / "melati_ensemble.py")
    new = module("new_ensemble", HERE / "hub" / "melati_ensemble.py")
    W, _, _, Y = load("eval")
    pick = np.linspace(0, len(W) - 1, a.n).round().astype(int)
    Wn = [W[i] for i in pick]
    res = {"old_revision": OLD, "clips_compared": a.n, "batch": a.batch, "frontends": frontends(new, Wn),
           "compare": {}}
    for bf16 in (False, True):
        tag = "bf16" if bf16 else "fp32"
        o = old.Ensemble(path, bf16=bf16)
        Po, so = run(o, Wn, a.batch)
        del o
        free()
        m = new.Ensemble(path, bf16=bf16)
        Pn, sn = run(m, Wn, a.batch)
        del m
        free()
        cmp = {"old_clips_per_s": round(a.n / so, 1), "new_clips_per_s": round(a.n / sn, 1), "members": {}}
        for k in Po:
            d = np.abs(Po[k] - Pn[k])
            cmp["members"][k] = {"max_abs_diff": round(float(d.max()), 6), "mean_abs_diff": float(d.mean())}
            print(f"[native] {tag} {k}: max |old - new| {d.max():.6f}, mean {d.mean():.2e}", flush=True)
        avg_o, avg_n = np.mean(list(Po.values()), 0), np.mean(list(Pn.values()), 0)
        cmp["ensemble_max_abs_diff"] = round(float(np.abs(avg_o - avg_n).max()), 6)
        print(f"[native] {tag} ensemble: max |old - new| {cmp['ensemble_max_abs_diff']:.6f}; "
              f"{cmp['old_clips_per_s']} -> {cmp['new_clips_per_s']} clips/s", flush=True)
        res["compare"][tag] = cmp
    OUT.write_text(json.dumps(res, indent=1))
    if a.full:
        ref = json.loads((RESULTS / "ensemble_fp32.json").read_text())
        res["full"] = {}
        for bf16 in (False, True):
            tag = "bf16" if bf16 else "fp32"
            m = new.Ensemble(path, bf16=bf16)
            P, s = run(m, W, 64)
            del m
            free()
            avg = np.mean(list(P.values()), 0)
            full = {"clips": int(len(W)), "clips_per_s_b64": round(len(W) / s, 1), "map": round(mean_ap(avg, Y), 4),
                    "members": {k: round(mean_ap(p, Y), 4) for k, p in P.items()}}
            print(f"[native] full {tag}: {full}", flush=True)
            res["full"][tag] = full
        res["release_map"] = {"fp32": ref["map"], "bf16": ref["map_bf16"],
                              "members": {k: v["map"] for k, v in ref["members"].items()}}
        for tag in ("fp32", "bf16"):
            assert abs(res["full"][tag]["map"] - res["release_map"][tag]) <= 1e-4, (tag, res["full"][tag]["map"])
        OUT.write_text(json.dumps(res, indent=1))
    print("NATIVE_OK", flush=True)


if __name__ == "__main__":
    main()
