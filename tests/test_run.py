import pytest

from melati import run

CLUSTER = {"front", "diar", "asr", "align", "mos", "tag"}             # serve/cluster.json: no sound tagger


def test_without_a_sound_tagger_the_sounds_stage_is_left_out(capsys):
    stages = ("front", "diar", "seg", "asr", "align", "quality", "tag", "sounds", "fill", "export")
    assert run.servable(stages, CLUSTER.__contains__) == tuple(s for s in stages if s != "sounds")
    assert "sounds stage is left out" in capsys.readouterr().out
    assert run.servable(stages, (CLUSTER | {"sounds"}).__contains__) == stages


def test_a_stage_without_its_service_stops_the_run_before_it_starts():
    with pytest.raises(SystemExit, match=r"mos \(stage quality\)"):
        run.servable(("asr", "quality", "sounds"), (CLUSTER - {"mos"}).__contains__)
    assert run.servable(("seg", "export"), set().__contains__) == ("seg", "export")
