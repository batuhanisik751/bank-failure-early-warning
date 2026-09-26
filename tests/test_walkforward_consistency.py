"""Rule 6.7 audit of a walk-forward artefact tree: every fitted year's tuning closed before
the year's first prediction date. Synthetic ``models/walkforward`` trees only; no data files.
"""

from __future__ import annotations

import json
from pathlib import Path

from bankcanary.evaluation import walkforward as w


def write_config(root: Path, year: int, model: str, config: dict) -> Path:
    path = root / str(year) / model / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config))
    return path


def tuned(validation_end: str | None) -> dict:
    return {"model": "logit", "horizon": 4, "tuning": {"validation_end": validation_end}}


def test_consistent_tree_passes_and_texas_is_skipped(tmp_path):
    write_config(tmp_path, 2010, "logit", tuned("2009-12-31"))
    write_config(tmp_path, 2010, "gbdt_8q", tuned("2007-12-31"))
    write_config(tmp_path, 2010, "texas", {"model": "texas", "fit": "none (ranking)"})
    (tmp_path / "latest").mkdir()  # a non-year directory is ignored
    assert w.check_tuning_consistency(tmp_path, 60) == []


def test_missing_and_late_tuning_are_flagged(tmp_path):
    write_config(tmp_path, 2010, "logit", tuned("2009-12-31"))
    no_tuning = write_config(tmp_path, 2010, "gbdt", {"model": "gbdt", "horizon": 4})
    late = write_config(tmp_path, 2011, "hazard", tuned("2011-06-30"))
    boundary = write_config(tmp_path, 2012, "logit", tuned("2012-05-30"))
    blank = write_config(tmp_path, 2012, "gbdt", tuned(None))
    flagged = w.check_tuning_consistency(tmp_path, lag_days=60)
    assert [(v["year"], v["model"]) for v in flagged] == [
        (2010, "gbdt"),
        (2011, "hazard"),
        (2012, "gbdt"),
        (2012, "logit"),
    ]
    by_path = {v["path"]: v["reason"] for v in flagged}
    assert by_path[str(no_tuning)] == "no tuning record"
    assert by_path[str(late)].startswith("validation_end 2011-06-30 >= first prediction 2011-05-30")
    assert "2012-05-30 >= first prediction 2012-05-30" in by_path[str(boundary)]
    assert by_path[str(blank)] == "tuning has no validation_end"


def test_lag_moves_the_first_prediction_date(tmp_path):
    write_config(tmp_path, 2012, "logit", tuned("2012-05-30"))
    assert w.check_tuning_consistency(tmp_path, lag_days=60)  # 05-30 is the boundary at 60 days
    assert w.check_tuning_consistency(tmp_path, lag_days=90) == []  # first prediction 06-29
