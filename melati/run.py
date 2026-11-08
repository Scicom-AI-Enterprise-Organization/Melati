"""Run the Melati pipeline over a source list, against a running cluster of model servers.

    python serve/launch.py up --config serve/cluster.json          # the GPUs: one server per model replica
    python -m melati.run --sources sources.jsonl --work $MELATI_DATA/podcast/work \\
        --out $MELATI_DATA/podcast/export --endpoints $MELATI_DATA/serve/endpoints.json

This process owns no GPU. It keeps --recordings recordings in flight, each walking every stage in
melati.pipeline in order, and does the CPU work (decoding, resampling, VAD, FLAC writing) in local
process and thread pools. Throughput is set by the servers: add replicas in serve/cluster.json (or
raise a vLLM service's tensor-parallel size) and run `launch.py up` again; nothing here changes.

`--stages` limits a run to some stages (e.g. `--stages front diar seg` to stop before ASR while the
ASR model is still training); a later run picks up from the files already written.
`--assemble` gathers every recording's export rows into parquet shards of the manifest.
"""
from __future__ import annotations

import os
import argparse
import asyncio
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from melati import pipeline as P

MELATI_DATA = os.environ.get("MELATI_DATA", "/root/melati/data")   # data, models, results: deploy/env.sh


async def main_async(args):
    from melati.client import Client, Endpoints
    recs = [json.loads(l) for l in args.sources.open()]
    if args.limit:
        recs = recs[: args.limit]
    stages = tuple(args.stages)
    todo = [r for r in recs if not all((args.work / r["id"] / P.OUTPUT[s]).exists() for s in stages)]
    print(f"[run] {len(todo)}/{len(recs)} recordings need {', '.join(stages)}", flush=True)
    for r in todo:
        (args.work / r["id"]).mkdir(parents=True, exist_ok=True)
    limits = dict(kv.split("=") for kv in args.limits) if args.limits else {}
    cfg = json.loads(args.export_config.read_text()) if args.export_config else {}
    with ProcessPoolExecutor(args.cpu_workers) as cpu, ThreadPoolExecutor(args.io_workers) as io:
        async with Client(Endpoints(args.endpoints), {k: int(v) for k, v in limits.items()}) as client:
            ctx = P.Ctx(work=args.work, client=client, cpu=cpu, io=io, allowed=tuple(args.allowed),
                        emotion_models=tuple(args.emotion_models), nv_models=tuple(args.nv_models),
                        nv_frame_models=tuple(args.nv_frame_models), out=args.out,
                        export_cfg=cfg)
            sem = asyncio.Semaphore(args.recordings)
            t0, done, failed = time.time(), 0, 0

            async def one(rec):
                nonlocal done, failed
                async with sem:
                    try:
                        await P.run_recording(ctx, rec, stages)
                        done += 1
                    except Exception as e:  # noqa: BLE001 - one recording must not stop the run
                        failed += 1
                        (args.work / rec["id"] / "error.txt").write_text(f"{type(e).__name__}: {e}")
                        print(f"[run] {rec['id']} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

            async def report():
                while True:
                    await asyncio.sleep(60)
                    el = time.time() - t0
                    per = {k: f"{v[1]} in {v[0] / max(v[1], 1):.1f}s avg" for k, v in ctx.timing.items()}
                    print(f"[run] {done} done, {failed} failed, {el / 60:.1f} min | {per}", flush=True)

            rep = asyncio.create_task(report())
            await asyncio.gather(*[one(r) for r in todo])
            rep.cancel()
            el = time.time() - t0
            hours = sum(json.loads((args.work / r["id"] / "sep.json").read_text())["duration_s"]
                        for r in todo if (args.work / r["id"] / "sep.json").exists()) / 3600
            summary = {"recordings": len(todo), "done": done, "failed": failed, "wall_s": round(el, 1),
                       "audio_h": round(hours, 3), "x_realtime": round(hours * 3600 / max(el, 1e-9), 1),
                       "stage_seconds": {k: round(v[0], 1) for k, v in ctx.timing.items()},
                       "stages": list(stages)}
            print(f"[run] {json.dumps(summary)}", flush=True)
            if args.timing:
                with args.timing.open("a") as fh:
                    fh.write(json.dumps(dict(summary, when=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")


def assemble(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    recs = [json.loads(l) for l in args.sources.open()]
    rows, part = [], 0
    (args.out / "manifest").mkdir(parents=True, exist_ok=True)
    for r in recs:
        p = args.work / r["id"] / "export.jsonl"
        rows += P.read_jsonl(p)
        if len(rows) >= 200_000:
            pq.write_table(pa.Table.from_pylist(rows), args.out / "manifest" / f"part-{part:04d}.parquet",
                           compression="zstd")
            rows, part = [], part + 1
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), args.out / "manifest" / f"part-{part:04d}.parquet",
                       compression="zstd")
    print(f"[assemble] {part + (1 if rows else 0)} parquet parts -> {args.out / 'manifest'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="export root (audio/ + manifest/)")
    ap.add_argument("--endpoints", nargs="+", type=Path, default=[Path(f"{MELATI_DATA}/serve/endpoints.json")],
                    help="one endpoints file, or several (services of separate clusters) merged")
    ap.add_argument("--stages", nargs="+", default=list(P.STAGES), choices=P.STAGES)
    ap.add_argument("--recordings", type=int, default=48, help="recordings in flight")
    ap.add_argument("--cpu-workers", type=int, default=48)
    ap.add_argument("--io-workers", type=int, default=32)
    ap.add_argument("--limits", nargs="*", default=None, help="per-service in-flight caps per replica, svc=N")
    ap.add_argument("--allowed", nargs="*", default=["ms", "en", "zh", "ta"])
    ap.add_argument("--emotion-models", nargs="*", default=["h_whisper", "h_whisper_ft", "h_w2vbert", "h_wavlm", "h_e2v", "meralion", "e2v"])
    ap.add_argument("--nv-models", nargs="*", default=["nv_whisper"],
                    help="non-verbal heads on the tag server; none served -> Qwen3-Omni (omni), if the cluster has it")
    ap.add_argument("--nv-frame-models", nargs="*", default=["nvf_whisper"],
                    help="frame-level non-verbal detector on the tag server (melati.nv_frames); none served -> skipped")
    ap.add_argument("--export-config", type=Path, default=None)
    ap.add_argument("--timing", type=Path, default=None, help="append the run summary here")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        return assemble(args)
    if "export" in args.stages and args.out is None:
        ap.error("--out is required when export is one of the stages")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
