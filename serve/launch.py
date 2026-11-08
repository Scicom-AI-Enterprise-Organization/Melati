#!/usr/bin/env python3
"""Start, stop and list the model servers of a Melati cluster, from one JSON config.

    python serve/launch.py up     --config serve/cluster.json      # start what is not running, wait for health
    python serve/launch.py status --config serve/cluster.json
    python serve/launch.py down   --config serve/cluster.json [--only asr]
    python serve/launch.py restart --config serve/cluster.json --only tag   # one replica at a time, under load

Every GPU model is a service; a service has replicas; a replica owns a list of GPUs. For a vLLM
service the length of that list is its tensor-parallel size, so scaling is a config edit:

    "asr":   {"kind": "vllm", "model": "...", "port": 8100, "replicas": [[0], [1]]}     two replicas, TP 1
    "diar":  {"kind": "vllm", "model": "...", "port": 8200, "replicas": [[2, 3]]}       one replica, TP 2
    "align": {"kind": "app", "app": "serve.align_server:app", "port": 8400, "replicas": [[4], [5]]}

Services may share a GPU (set `gpu_memory_utilization` for vLLM ones so the others fit). Replicas
get consecutive ports; a replica set to null is retired and keeps its port unused. `up` writes the endpoints file every client reads ({service: [url, ...]}),
listing only replicas that answered their health check.

Paths in a config are written ${MELATI_DATA}/... and ${MELATI_ROOT}/... (deploy/env.sh), filled in from the environment
when it is read (`load`), so one config serves any machine.
"""
import argparse, json, os, re, signal, subprocess, sys, time  # noqa: E401
from pathlib import Path
from urllib.request import urlopen

MELATI_ROOT = os.environ.get("MELATI_ROOT", "/root/melati")        # environments, checkouts: deploy/env.sh

REPO = Path(__file__).resolve().parent.parent


def load(path):
    """A cluster config with the environment's variables filled into its strings. MELATI_ROOT and MELATI_DATA default
    as in deploy/env.sh, and are left set for the servers it starts; a variable nobody set stops it here, before a
    server writes to a path named after the variable."""
    os.environ.setdefault("MELATI_ROOT", MELATI_ROOT)
    os.environ.setdefault("MELATI_DATA", f"{os.environ['MELATI_ROOT']}/data")

    def fill(o):
        if isinstance(o, dict):
            return {k: fill(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fill(v) for v in o]
        if isinstance(o, str):
            s = os.path.expandvars(o)
            if re.search(r"\$\{?[A-Za-z_]", s):
                sys.exit(f"{path}: {o!r} names a variable that is not set")
            return s
        return o
    return fill(json.loads(Path(path).read_text()))


def replicas(cfg):
    """Every replica as (service, its config, number, GPUs, port). A replica set to null is retired: nothing runs
    there, and its port stays reserved so the replicas after it keep theirs (and a running cluster keeps its URLs)."""
    for name, svc in cfg["services"].items():
        for k, gpus in enumerate(svc["replicas"]):
            if gpus is not None:
                yield name, svc, k, gpus, svc["port"] + k


def _get(port, path):
    try:
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
            return r.status, r.read()
    except Exception:
        return None, b""


def healthy(port, svc=None, name=None):
    """OUR service answers on the port: a vLLM server lists its served name in /v1/models, an app's
    /health is JSON with ok=true. A bare 200 is not enough: on the shared box, 9100 and 9400 are
    metrics exporters that answer anything."""
    if svc is not None and svc.get("kind") == "vllm":
        code, body = _get(port, "/v1/models")
        return code == 200 and svc.get("served_name", f"melati-{name}").encode() in body
    code, body = _get(port, "/health")
    if code != 200:
        return False
    try:
        return bool(json.loads(body).get("ok", True)) if body.strip().startswith(b"{") else False
    except ValueError:
        return False


def alive(pidf: Path) -> bool:
    """The replica's recorded process still runs: a server loading its models is not yet healthy, and a watchdog's
    `up` must not start a second copy of it on the same GPU."""
    try:
        os.kill(int(pidf.read_text()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def port_taken(port):
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def command(name, svc, gpus, port):
    venv = svc.get("venv", f"{MELATI_ROOT}/.venv-melati")
    if svc["kind"] == "vllm":
        cmd = [f"{venv}/bin/vllm", "serve", svc["model"], "--host", "127.0.0.1", "--port", str(port),
               "--served-model-name", svc.get("served_name", f"melati-{name}"),
               "--tensor-parallel-size", str(len(gpus)),
               "--gpu-memory-utilization", str(svc.get("gpu_memory_utilization", 0.85))]
        return cmd + [str(a) for a in svc.get("args", [])]
    return [f"{venv}/bin/uvicorn", svc["app"], "--host", "127.0.0.1", "--port", str(port),
            "--workers", "1", "--log-level", "warning"] + [str(a) for a in svc.get("args", [])]


def start(cfg, name, svc, k, gpus, port, tag="up"):
    run = Path(cfg["run_dir"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
               PYTHONPATH=f"{REPO}:{svc.get('pythonpath', '')}".rstrip(":"))
    env.update({k_: str(v) for k_, v in cfg.get("env", {}).items()})
    env.update({k_: str(v) for k_, v in svc.get("env", {}).items()})
    log = open(run / f"{name}-{k}.log", "ab")  # noqa: SIM115 - the child keeps it
    p = subprocess.Popen(command(name, svc, gpus, port), cwd=svc.get("cwd", str(REPO)), env=env,
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (run / f"{name}-{k}.pid").write_text(str(p.pid))
    print(f"[{tag}] {name}#{k} gpus={gpus} :{port} pid {p.pid}")
    return p


def wait(cfg, name, k, port, p, deadline, tag="up"):
    svc = cfg["services"][name]
    while not healthy(port, svc, name):
        if p.poll() is not None:
            print(f"[{tag}] {name}#{k} EXITED ({p.returncode}); see {cfg['run_dir']}/{name}-{k}.log")
            return False
        if time.time() > deadline:
            print(f"[{tag}] {name}#{k} not healthy after the startup timeout")
            return False
        time.sleep(3)
    print(f"[{tag}] {name}#{k} healthy on :{port}")
    return True


def up(cfg, only=None):
    sys.stdout.reconfigure(line_buffering=True)
    run = Path(cfg["run_dir"]); run.mkdir(parents=True, exist_ok=True)
    started = []
    todo = []
    for name, svc, k, gpus, port in replicas(cfg):
        if only and name not in only:
            continue
        if healthy(port, svc, name):
            print(f"[up] {name}#{k} already healthy on :{port}")
            continue
        if alive(run / f"{name}-{k}.pid"):
            print(f"[up] {name}#{k} still starting (its process is alive); left alone")
            continue
        if port_taken(port):
            print(f"[up] {name}#{k} NOT started: :{port} is held by something else")
            continue
        todo.append((name, svc, k, gpus, port))
    deadline = time.time() + cfg.get("startup_timeout_s", 1200)
    # vLLM replicas one at a time: vLLM sizes its KV cache by profiling how much memory the GPU has
    # left, and another server loading on the same GPU at that moment makes it find none
    # ("No available memory for the cache blocks"). The plain apps then start together.
    for name, svc, k, gpus, port in [t for t in todo if t[1]["kind"] == "vllm"]:
        wait(cfg, name, k, port, start(cfg, name, svc, k, gpus, port), deadline)
    for name, svc, k, gpus, port in [t for t in todo if t[1]["kind"] != "vllm"]:
        started.append((name, k, port, start(cfg, name, svc, k, gpus, port)))
    for name, k, port, p in started:
        wait(cfg, name, k, port, p, deadline)
    write_endpoints(cfg)


def restart(cfg, only=None, which=None):
    """Roll replicas onto new code one at a time, each healthy again before the next goes down, so a running pipeline
    keeps going: its client retries the calls the restarting replica dropped on the others."""
    sys.stdout.reconfigure(line_buffering=True)
    run = Path(cfg["run_dir"])
    for name, svc, k, gpus, port in replicas(cfg):
        if (only and name not in only) or (which is not None and k not in which):
            continue
        pidf = run / f"{name}-{k}.pid"
        old = pidf.read_text() if pidf.exists() else ""
        pidf.write_text(str(os.getpid()))        # until the new one starts, a watchdog's `up` sees it as starting
        old_pidf = run / f"{name}-{k}.pid.old"
        old_pidf.write_text(old)
        try:
            os.killpg(int(old), signal.SIGTERM)
        except (ValueError, ProcessLookupError):
            pass
        t = time.time()
        while (alive(old_pidf) or port_taken(port)) and time.time() - t < 180:
            time.sleep(2)
        old_pidf.unlink()
        if port_taken(port):
            print(f"[restart] {name}#{k} did not stop in 180 s; left alone")
            pidf.write_text(old)
            continue
        p = start(cfg, name, svc, k, gpus, port, "restart")
        if not wait(cfg, name, k, port, p, time.time() + cfg.get("startup_timeout_s", 1200), "restart"):
            print(f"[restart] stopping the roll: {name}#{k} did not come back")
            break
    write_endpoints(cfg)


def write_endpoints(cfg):
    ends = {}
    for name, svc, k, gpus, port in replicas(cfg):
        if healthy(port, svc, name):
            ends.setdefault(name, []).append(f"http://127.0.0.1:{port}")
    path = Path(cfg["endpoints"])
    tmp = path.with_suffix(f".{os.getpid()}.part")        # a watchdog's `up` may be writing it at the same moment
    tmp.write_text(json.dumps(ends, indent=1))
    tmp.replace(path)
    print(f"[endpoints] {json.dumps(ends)} -> {path}")


def down(cfg, only=None):
    run = Path(cfg["run_dir"])
    for name, svc, k, gpus, port in replicas(cfg):
        if only and name not in only:
            continue
        pidf = run / f"{name}-{k}.pid"
        if pidf.exists():
            try:
                os.killpg(int(pidf.read_text()), signal.SIGTERM)    # the whole session: vLLM forks workers
                print(f"[down] {name}#{k} pid {pidf.read_text()}")
            except ProcessLookupError:
                pass
            pidf.unlink()
    time.sleep(5)
    write_endpoints(cfg)


def status(cfg):
    for name, svc, k, gpus, port in replicas(cfg):
        print(f"{name}#{k:<2} gpus={str(gpus):<10} :{port}  {'healthy' if healthy(port, svc, name) else 'DOWN'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["up", "down", "restart", "status"])
    ap.add_argument("--config", type=Path, default=REPO / "serve" / "cluster.json")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--replicas", nargs="*", type=int, default=None, help="restart: only these replica numbers")
    args = ap.parse_args()
    cfg = load(args.config)
    {"up": lambda: up(cfg, args.only), "down": lambda: down(cfg, args.only), "restart": lambda: restart(cfg, args.only, args.replicas),
     "status": lambda: status(cfg)}[args.action]()


if __name__ == "__main__":
    sys.exit(main())
