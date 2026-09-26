"""Copy the latest walk-forward models into ``models/production/`` (CONTRACT section 15).

``uv run python scripts/promote_production_models.py [--year Y] [--models gbdt_mono hazard]``
copies ``pipeline.joblib``, ``features.json`` and ``config.json`` from
``models/walkforward/<Y>/<model>/`` to ``models/production/<model>/`` and writes
``model_version.json`` beside them. Nothing is refitted and nothing is read from git: the
version is the config's own ``model_version`` (``<model>-<train_end_repdte>-<model hash>``,
stamped by ``walkforward.save_year`` or ``scripts/stamp_model_versions.py`` from the fitted
estimator itself), so it changes exactly when the model does. ``calibration.joblib`` is
copied from the walk-forward year only when the production directory holds no production
map of its own (the flat ``calibration.json`` that ``calibration.production_calibrator``
writes beside its map); an existing production map is kept, since it is fitted on later
years than the year's own walk-forward map. ``trained_at`` is the pipeline's modification
time unless ``--trained-at`` overrides it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS: tuple[str, ...] = ("gbdt_mono", "hazard")
ARTEFACTS: tuple[str, ...] = (
    "pipeline.joblib",
    "calibration.joblib",
    "features.json",
    "config.json",
)
VERSION_FILE = "model_version.json"
CALIBRATION_RECORD = "calibration.json"


def latest_year(walkforward_dir: Path) -> int:
    """The largest ``<Y>`` directory under ``models/walkforward/``; raises when there is none."""
    years = [int(p.name) for p in walkforward_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not years:
        raise FileNotFoundError(f"no walk-forward years under {walkforward_dir}")
    return max(years)


def has_production_map(production_model_dir: Path) -> bool:
    """True when the directory holds a ``production_calibrator`` map: ``calibration.joblib``
    beside a flat ``calibration.json`` (``calibration_years`` at top level, no walk-forward
    ``config`` block)."""
    record = production_model_dir / CALIBRATION_RECORD
    if not record.exists() or not (production_model_dir / "calibration.joblib").exists():
        return False
    try:
        meta = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and "config" not in meta and "calibration_years" in meta


def artefact_time(artefact: Path) -> str:
    """The artefact's modification time as an ISO-8601 UTC timestamp."""
    return datetime.fromtimestamp(artefact.stat().st_mtime, tz=UTC).isoformat()


def read_version(config: dict, config_path: Path) -> tuple[str, str]:
    """``(model_version, model_hash)`` from a stamped config; raises when it is unstamped."""
    try:
        return str(config["model_version"]), str(config["model_hash"])
    except KeyError as exc:
        raise KeyError(
            f"{config_path} carries no model_version/model_hash; run "
            "scripts/stamp_model_versions.py (or refit) before promoting"
        ) from exc


def promote_model(
    model: str,
    year: int,
    walkforward_dir: Path,
    production_dir: Path,
    horizon: int = 4,
    trained_at: str | None = None,
    repo: Path = ROOT,
) -> dict:
    """Copy one model's artefacts and write its ``model_version.json``; returns that record."""
    suffix = "" if int(horizon) == 4 else f"_{int(horizon)}q"
    src = walkforward_dir / str(int(year)) / f"{model}{suffix}"
    missing = [name for name in ARTEFACTS if not (src / name).exists()]
    if missing:
        raise FileNotFoundError(f"{src} lacks {', '.join(missing)}")
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    version, digest = read_version(config, src / "config.json")
    dst = production_dir / model
    dst.mkdir(parents=True, exist_ok=True)
    keep_map = has_production_map(dst)
    for name in ARTEFACTS:
        if keep_map and name == "calibration.joblib":
            continue
        shutil.copyfile(src / name, dst / name)
    if keep_map:
        years = json.loads((dst / CALIBRATION_RECORD).read_text(encoding="utf-8"))
        map_note = (
            "calibration.joblib is the production map fitted on test years "
            f"{years.get('calibration_years')} (kept, not the walk-forward year's map)."
        )
    else:
        map_note = "calibration.joblib is that walk-forward year's own map."
    record = {
        "model_version": version,
        "model": model,
        "model_hash": digest,
        "train_end_repdte": config["train_repdte_max"],
        "trained_at": trained_at or artefact_time(src / "pipeline.joblib"),
        "features_version": config["features_version"],
        "horizon": int(config["horizon"]),
        "walkforward_year": int(config["test_year"]),
        "source": str(src.relative_to(repo)) if src.is_relative_to(repo) else str(src),
        "provenance": "override" if trained_at else "artefact",
        "calibration": "production" if keep_map else "walkforward",
        "notes": (
            f"Walk-forward {model} for test year {config['test_year']}, trained on reports "
            f"{config['train_repdte_min']}..{config['train_repdte_max']} "
            f"({config['n_train']:,} rows, {config['positives_train']:,} failures); "
            f"version hash from the fitted estimator; {map_note}"
        ),
    }
    (dst / VERSION_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=None, help="walk-forward year, default latest")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--trained-at", default=None, help="override the training timestamp")
    args = parser.parse_args(argv)
    walkforward_dir = args.models_dir / "walkforward"
    year = args.year if args.year is not None else latest_year(walkforward_dir)
    for model in args.models:
        record = promote_model(
            model, year, walkforward_dir, args.models_dir / "production", args.horizon,
            args.trained_at,
        )  # fmt: skip
        print(
            f"{record['model_version']}: {record['source']} -> models/production/{model} "
            f"(trained_at {record['trained_at']}, {record['provenance']}, "
            f"{record['calibration']} map)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
