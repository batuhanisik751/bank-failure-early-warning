"""``scripts/promote_production_models.py`` on a synthetic walk-forward tree: no data, no git."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_models_p2_adversarial import load_script

promote = load_script("promote_production_models")
TRAINED_AT = "2026-01-01T00:00:00+00:00"


def make_tree(root: Path, year: int, model: str, train_end: str, stamped: bool = True) -> Path:
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
    if stamped:
        digest = f"{model}{year}".encode().hex().ljust(40, "0")[:40]
        config.update(model_hash=digest, model_version=f"{model}-{train_end}-{digest[:7]}")
    (src / "config.json").write_text(json.dumps(config))
    return src


def _promote(root: Path, model: str, year: int = 2024, **kw) -> dict:
    return promote.promote_model(
        model,
        year,
        root / "models" / "walkforward",
        root / "models" / "production",
        repo=root,
        **kw,
    )


def test_latest_year_and_missing_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="no walk-forward years"):
        promote.latest_year(tmp_path)
    make_tree(tmp_path, 2019, "gbdt_mono", "2017-12-31")
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    (tmp_path / "models" / "walkforward" / "notes").mkdir()
    assert promote.latest_year(tmp_path / "models" / "walkforward") == 2024
    (tmp_path / "models" / "walkforward" / "2024" / "gbdt_mono" / "calibration.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="calibration.joblib"):
        _promote(tmp_path, "gbdt_mono", trained_at=TRAINED_AT)


def test_promote_copies_artefacts_and_writes_the_version_from_the_config(tmp_path):
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    make_tree(tmp_path, 2024, "hazard", "2023-12-31")
    production = tmp_path / "models" / "production"
    records = [_promote(tmp_path, m, trained_at=TRAINED_AT) for m in ("gbdt_mono", "hazard")]
    expected = {"gbdt_mono": "2022-12-31", "hazard": "2023-12-31"}
    for model, record in zip(expected, records, strict=True):
        config = json.loads(
            (tmp_path / "models" / "walkforward" / "2024" / model / "config.json").read_text()
        )
        assert record["model_version"] == config["model_version"]
        assert record["model_version"].startswith(f"{model}-{expected[model]}-")
        assert record["model_hash"] == config["model_hash"] and "git_sha" not in record
        out = production / model
        assert {p.name for p in out.iterdir()} == set(promote.ARTEFACTS) | {promote.VERSION_FILE}
        assert (out / "pipeline.joblib").read_bytes() == b"pipe-" + model.encode()
        assert (out / "calibration.joblib").read_bytes() == b"cal-" + model.encode()
        saved = json.loads((out / promote.VERSION_FILE).read_text())
        assert saved == record
        assert saved["provenance"] == "override" and saved["trained_at"].startswith("2026-01-01")
        assert saved["calibration"] == "walkforward" and "year's own map" in saved["notes"]
        assert saved["source"] == f"models/walkforward/2024/{model}"
        assert saved["features_version"] == "v2" and saved["walkforward_year"] == 2024
        assert "1,000 rows" in saved["notes"]
    # idempotent: a second promotion rewrites the same bytes
    before = (production / "gbdt_mono" / promote.VERSION_FILE).read_bytes()
    _promote(tmp_path, "gbdt_mono", trained_at=TRAINED_AT)
    assert (production / "gbdt_mono" / promote.VERSION_FILE).read_bytes() == before


def test_promote_keeps_an_existing_production_map_and_refuses_unstamped_configs(tmp_path):
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    make_tree(tmp_path, 2024, "hazard", "2023-12-31", stamped=False)
    production = tmp_path / "models" / "production" / "gbdt_mono"
    production.mkdir(parents=True)
    (production / "calibration.joblib").write_bytes(b"production-map")
    (production / "calibration.json").write_text(json.dumps({"calibration_years": [2023, 2024]}))
    record = _promote(tmp_path, "gbdt_mono")
    assert (production / "calibration.joblib").read_bytes() == b"production-map"
    assert record["calibration"] == "production" and "[2023, 2024]" in record["notes"]
    assert record["provenance"] == "artefact" and record["trained_at"].endswith("+00:00")
    assert (production / "pipeline.joblib").read_bytes() == b"pipe-gbdt_mono"
    with pytest.raises(KeyError, match="stamp_model_versions"):
        _promote(tmp_path, "hazard")
    assert not (tmp_path / "models" / "production" / "hazard" / promote.VERSION_FILE).exists()


def test_main_promotes_the_latest_year_by_default(tmp_path, capsys):
    make_tree(tmp_path, 2023, "gbdt_mono", "2021-12-31")
    make_tree(tmp_path, 2024, "gbdt_mono", "2022-12-31")
    make_tree(tmp_path, 2024, "hazard", "2023-12-31")
    argv = ["--models-dir", str(tmp_path / "models"), "--trained-at", TRAINED_AT]
    assert promote.main(argv) == 0
    out = capsys.readouterr().out
    assert "gbdt_mono-2022-12-31-" in out and "hazard-2023-12-31-" in out
    assert "walkforward map" in out
    assert (tmp_path / "models" / "production" / "hazard" / "model_version.json").exists()
