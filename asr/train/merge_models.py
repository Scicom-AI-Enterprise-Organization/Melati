#!/usr/bin/env python3
"""Merge fine-tunes of whisper-large-v3 by their weight deltas (task vectors). RUNS ON THE BOX.

Each run was trained for one thing: replay keeps languages, the language KL or label keeps identification, the
Indic arm adds languages Common Voice lacks, a full fine-tune at a small learning rate barely loops. A merge asks
whether one model can keep all of it.

A LoRA run's delta is rebuilt exactly from its fp32 adapter (B @ A * alpha / r). Its merged/ checkpoint was
saved in bfloat16, where an update smaller than half a unit in the last place of the weight it is added to
rounds away, so (merged - base) is not the delta that was trained; --check prints how far apart they are. A full
fine-tune has no adapter, so its delta is (merged - base), with that rounding in it.

  mean   base + sum_i w_i d_i / sum_i w_i                         (a soup over deltas)
  sum    base + lam * sum_i w_i d_i                               (task arithmetic)
  ties   per tensor, keep each delta's top `density` share by magnitude, elect a sign per entry by the summed
         mass, average the kept entries that agree with it; base + lam * that       (Yadav et al., 2023)
  dare   drop each delta entry with probability 1 - density and rescale by 1/density, then ties' election
                                                                                    (Yu et al., 2024)

The output is a run directory (<out>/merged/ in float16, plus run.json), so asr/train/eval_run.sh scores it like
any other run.

    python asr/train/merge_models.py --method mean --out $MELATI_DATA/asr/merges/soup25 \\
        --runs $MELATI_DATA/asr/runs2/v3_lora_r64_all_cv22_r25_tts_lr5e-4_s1000_liddistill \\
               $MELATI_DATA/asr/runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000_lidlabel
"""
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_ID = "openai/whisper-large-v3"


def base_file():
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(BASE_ID, allow_patterns=["model.safetensors", "generation_config.json"]))
    return d / "model.safetensors", d / "generation_config.json"


class LoraDeltas:
    """A run's LoRA updates, each built on demand as B @ A * alpha / r in fp32 (all of them at once would be
    gigabytes per run)."""

    def __init__(self, a, b, scale):
        self.a, self.b, self.scale = a, b, scale

    def __contains__(self, name):
        return name in self.a

    def get(self, name):
        return (self.b[name] @ self.a[name]) * self.scale if name in self.a else None

    def __getitem__(self, name):
        return self.get(name)


def lora_deltas(run: Path):
    """A run's LoraDeltas from its PEFT adapter, or None when the run has no adapter."""
    ad = run / "adapter"
    if not (ad / "adapter_config.json").exists():
        return None
    cfg = json.loads((ad / "adapter_config.json").read_text())
    assert not cfg.get("use_dora") and not cfg.get("use_rslora"), f"{run.name}: only plain LoRA is handled"
    scale = cfg["lora_alpha"] / cfg["r"]
    a, b = {}, {}
    with safe_open(str(ad / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            name = k.replace("base_model.model.", "", 1)
            if ".lora_A." in name:
                a[name.replace(".lora_A.weight", ".weight")] = f.get_tensor(k).float()
            elif ".lora_B." in name:
                b[name.replace(".lora_B.weight", ".weight")] = f.get_tensor(k).float()
            else:
                raise ValueError(f"{run.name}: unexpected adapter tensor {k}")
    assert a.keys() == b.keys()
    return LoraDeltas(a, b, scale)


class Weights:
    """Tensors of a saved checkpoint by name, single-file or sharded (model.safetensors.index.json)."""

    def __init__(self, d: Path):
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
            files = {f: safe_open(str(d / f), "pt") for f in set(wm.values())}
            self.where = {k: files[f] for k, f in wm.items()}
        else:
            f = safe_open(str(d / "model.safetensors"), "pt")
            self.where = {k: f for k in f.keys()}

    def get_tensor(self, name):
        return self.where[name].get_tensor(name)


def ties(ds, density, rng=None, dare=False):
    """Merge a list of same-shaped deltas the TIES way (optionally DARE's random drop first)."""
    kept = []
    for d in ds:
        if dare:
            m = (torch.rand(d.shape, generator=rng) < density).float()
            d = d * m / density
        else:
            k = max(1, int(density * d.numel()))
            thr = d.abs().flatten().kthvalue(d.numel() - k + 1).values if k < d.numel() else d.abs().min()
            d = torch.where(d.abs() >= thr, d, torch.zeros_like(d))
        kept.append(d)
    st = torch.stack(kept)
    sign = torch.sign(st.sum(0))
    agree = (torch.sign(st) == sign) & (st != 0)
    return (st * agree).sum(0) / agree.sum(0).clamp(min=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--method", choices=["mean", "sum", "ties", "dare"], required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="also report how far each bf16-saved delta is from the exact one")
    args = ap.parse_args()
    w = args.weights or [1.0] * len(args.runs)
    assert len(w) == len(args.runs)
    bfile, bgen = base_file()
    lora = [lora_deltas(r) for r in args.runs]
    full = [None if l is not None else Weights(r / "merged") for r, l in zip(args.runs, lora)]
    rng = torch.Generator().manual_seed(args.seed)
    out, report = {}, {}
    with safe_open(str(bfile), "pt") as fb:
        for name in fb.keys():
            base = fb.get_tensor(name).float()
            ds = []
            for i, r in enumerate(args.runs):
                if lora[i] is not None:
                    d = lora[i].get(name)
                    ds.append(torch.zeros_like(base) if d is None else d)
                else:
                    ds.append(full[i].get_tensor(name).float() - base)
            if all(not torch.any(d) for d in ds):
                out[name] = base.half()
                continue
            if args.method == "mean":
                m = sum(wi * d for wi, d in zip(w, ds)) / sum(w)
            elif args.method == "sum":
                m = args.lam * sum(wi * d for wi, d in zip(w, ds))
            else:
                m = args.lam * ties([wi * d for wi, d in zip(w, ds)], args.density, rng, dare=args.method == "dare")
            out[name] = (base + m).half()
            if args.check:
                for i, r in enumerate(args.runs):
                    if lora[i] is not None and name in lora[i]:
                        saved = Weights(r / "merged").get_tensor(name).float() - base
                        exact = lora[i].get(name)
                        rep = report.setdefault(r.name, [0.0, 0.0])
                        rep[0] += float(((saved - exact) ** 2).sum()); rep[1] += float((exact ** 2).sum())
    dst = args.out / "merged"
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    src = args.runs[0] / "merged"
    for f in src.iterdir():                      # config, tokenizer, processor: identical across runs
        if f.name not in ("model.safetensors", "generation_config.json"):
            shutil.copy2(f, dst / f.name)
    shutil.copy2(bgen, dst / "generation_config.json")
    meta = {"method": args.method, "runs": [str(r) for r in args.runs], "weights": w, "lam": args.lam,
            "density": args.density, "seed": args.seed, "base_model": BASE_ID, "dtype": "float16",
            "lora_deltas": "exact, from fp32 adapters", "bf16_save_delta_rel_error": None}
    if report:   # ||(saved merged - base) - exact delta|| / ||exact delta||, over every LoRA-updated weight
        meta["bf16_save_delta_rel_error"] = {k: round((e / max(t, 1e-30)) ** 0.5, 4) for k, (e, t) in report.items()}
    (args.out / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
