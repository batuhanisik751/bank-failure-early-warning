"""``scripts/promote_production_models.py`` on a synthetic walk-forward tree: no data, no git."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_models_p2_adversarial import load_script

promote = load_script("promote_production_models")


def make_tree(root: Path, year: int, model: str, train_end: str) -> Path:
    """A fake ``models/walkforward/<year>/<model>/`` with the four artefacts."""
    src = root / "models" / "walkforward" / str(year) / model
    src.mkdir(parents=True)
    (src / "pipeline.joblib").write_bytes(b"pipe-" + model.encode())
    (src / "calibration.joblib").write_bytes(b"cal-" + model.encode())
    (src / "features.json").write_text(json.dumps(["a", "b"]))
    config = {
        "model": model,
        "features_version": "v2",
        "horizon": 4,
        "test_year": year,
        "train_repdte_min": "2001-03-31",
        "train_repdte_max": train_end,
        "n_train": 1000,
        "positives_train": 12,
    }
    (src / "config.json").write_text(json.dumps(config))
    return src


def test_latest_year_and_missing_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="no walk-forward years"):
        promote.latest_year(tmp_path)
    make_tree(tmp_path, 2019, "gbdt_mono", "2017-12-31")
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    (tmp_path / "models" / "walkforward" / "notes").mkdir()
    assert promote.latest_year(tmp_path / "models" / "walkforward") == 2024
    (tmp_path / "models" / "walkforward" / "2024" / "gbdt_mono" / "calibration.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="calibration.joblib"):
        promote.promote_model(
            "gbdt_mono",
            2024,
            tmp_path / "models" / "walkforward",
            tmp_path / "models" / "production",
            tmp_path / "runs",
            git_sha="abc1234",
            trained_at="2026-01-01T00:00:00+00:00",
        )


def test_promote_copies_artefacts_and_writes_the_version_from_the_config(tmp_path):
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    make_tree(tmp_path, 2024, "hazard", "2023-12-31")
    walkforward, production = (
        tmp_path / "models" / "walkforward",
        tmp_path / "models" / "production",
    )
    records = [
        promote.promote_model(
            model,
            2024,
            walkforward,
            production,
            tmp_path / "runs",
            git_sha="abc1234",
            trained_at="2026-01-01T00:00:00+00:00",
            repo=tmp_path,
        )
        for model in ("gbdt_mono", "hazard")
    ]
    assert records[0]["model_version"] == "gbdt_mono-2022-12-31-abc1234"
    assert records[1]["model_version"] == "hazard-2023-12-31-abc1234"
    for model, record in zip(("gbdt_mono", "hazard"), records, strict=True):
        out = production / model
        assert {p.name for p in out.iterdir()} == set(promote.ARTEFACTS) | {promote.VERSION_FILE}
        assert (out / "pipeline.joblib").read_bytes() == b"pipe-" + model.encode()
        saved = json.loads((out / promote.VERSION_FILE).read_text())
        assert saved == record
        assert saved["provenance"] == "override" and saved["trained_at"].startswith("2026-01-01")
        assert saved["source"] == f"models/walkforward/2024/{model}"
        assert saved["features_version"] == "v2" and saved["walkforward_year"] == 2024
        assert "1,000 rows" in saved["notes"]
    # idempotent: a second promotion rewrites the same bytes
    before = (production / "gbdt_mono" / promote.VERSION_FILE).read_bytes()
    promote.promote_model(
        "gbdt_mono",
        2024,
        walkforward,
        production,
        tmp_path / "runs",
        git_sha="abc1234",
        trained_at="2026-01-01T00:00:00+00:00",
        repo=tmp_path,
    )
    assert (production / "gbdt_mono" / promote.VERSION_FILE).read_bytes() == before


def test_training_commit_falls_back_to_head_and_the_artefact_mtime(tmp_path):
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    artefact = tmp_path / "models" / "walkforward" / "2024" / "gbdt_mono" / "pipeline.joblib"
    sha, when, source = promote.training_commit(tmp_path / "runs" / "nowhere", artefact, tmp_path)
    assert source == "head" and sha and when.endswith("+00:00")
    assert (
        promote.model_version("gbdt_mono", "2022-12-31", "abc1234")
        == "gbdt_mono-2022-12-31-abc1234"
    )


def test_main_promotes_the_latest_year_by_default(tmp_path, capsys):
    make_tree(tmp_path, 2023, "gbdt_mono", "2021-12-31")
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    make_tree(tmp_path, 2024, "hazard", "2023-12-31")
    argv = [
        "--models-dir",
        str(tmp_path / "models"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--git-sha",
        "abc1234",
        "--trained-at",
        "2026-01-01T00:00:00+00:00",
    ]
    assert promote.main(argv) == 0
    out = capsys.readouterr().out
    assert "gbdt_mono-2022-12-31-abc1234" in out and "hazard-2023-12-31-abc1234" in out
    assert (tmp_path / "models" / "production" / "hazard" / "model_version.json").exists()
