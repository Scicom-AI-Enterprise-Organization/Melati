import json
import re
from pathlib import Path

from melati import pins

ROOT = Path(__file__).resolve().parents[1]
SETUP = sorted(ROOT.glob("deploy/box_*setup*.sh"))


def test_every_pin_is_a_full_commit():
    for name, rev in {**pins.HUB, **pins.GIT}.items():
        assert re.fullmatch(r"[0-9a-f]{40}", rev), name
    url = "https://github.com/fschmid56/PretrainedSED"
    assert pins.rev(url + ".git") == pins.GIT[url]
    assert pins.rev("/a/local/folder") is None


def test_the_cluster_configs_serve_every_vllm_model_at_its_pin():
    for path in sorted(ROOT.glob("serve/cluster*.json")):
        for name, svc in json.loads(path.read_text())["services"].items():
            if svc.get("kind") == "vllm":
                args = [str(a) for a in svc.get("args") or []]
                for flag in ("--revision", "--tokenizer-revision"):
                    assert flag in args, (path.name, name, flag)
                    assert args[args.index(flag) + 1] == pins.rev(svc["model"]), (path.name, name, flag)


def test_the_setup_scripts_download_and_check_out_pinned_revisions_only():
    for path in SETUP:
        text = path.read_text()
        assert "git clone" not in text, path.name                    # deploy/env.sh's checkout() pins the commit
        for url in re.findall(r"^checkout (\S+)", text, re.M):
            assert pins.rev(url), (path.name, url)
        for line in text.splitlines():
            if re.search(r"\bhf download\b|\$HF download", line):
                assert "--revision" in line, (path.name, line)
        for loop in re.findall(r"for repo in (.*?); do", text, re.S):
            for repo in loop.replace("\\", " ").split():
                assert pins.rev(repo), (path.name, repo)
        for name in re.findall(r"\$\(pin ([\w./-]+)\)", text):
            assert pins.rev(name), (path.name, name)
