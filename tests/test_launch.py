import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))
import launch  # noqa: E402


def test_a_retired_replica_keeps_its_port_and_runs_nothing():
    cfg = {"services": {"align": {"port": 18400, "replicas": [None, [1], [5]]}}}
    assert [(k, g, p) for _, _, k, g, p in launch.replicas(cfg)] == [(1, [1], 18401), (2, [5], 18402)]


def test_a_config_names_its_paths_by_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("MELATI_DATA", "/data/melati")
    monkeypatch.delenv("MELATI_ROOT", raising=False)
    cfg = tmp_path / "cluster.json"
    cfg.write_text('{"run_dir": "${MELATI_DATA}/serve", "services": {"mos": {"venv": "${MELATI_ROOT}/.venv-melati", '
                   '"args": ["--path", "${MELATI_DATA}"], "port": 1, "replicas": [[0]]}}}')
    got = launch.load(cfg)
    assert got["run_dir"] == "/data/melati/serve"
    assert got["services"]["mos"]["venv"] == "/root/melati/.venv-melati"
    assert got["services"]["mos"]["args"] == ["--path", "/data/melati"]


def test_a_config_naming_an_unset_variable_stops(tmp_path, monkeypatch):
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    cfg = tmp_path / "cluster.json"
    cfg.write_text('{"run_dir": "${NOT_SET_ANYWHERE}/serve", "services": {}}')
    try:
        launch.load(cfg)
    except SystemExit as e:
        assert "NOT_SET_ANYWHERE" in str(e)
    else:
        raise AssertionError("an unset variable was let through")
