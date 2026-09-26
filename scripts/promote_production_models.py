"""Copy the latest walk-forward models into ``models/production/`` (CONTRACT section 15).

``uv run python scripts/promote_production_models.py [--year Y] [--models gbdt_mono hazard]``
copies ``pipeline.joblib``, ``calibration.joblib``, ``features.json`` and ``config.json``
from ``models/walkforward/<Y>/<model>/`` to ``models/production/<model>/`` and writes
``model_version.json`` beside them. Nothing is refitted and nothing is read from the clock:
``train_end_repdte`` is the walk-forward config's ``train_repdte_max``; ``git_sha`` and
``trained_at`` come from the git commit that first recorded the model's walk-forward run
(``runs/walkforward/<run_id>/``, whose id is a hash of that same config), falling back to
``HEAD`` and the artefact's own modification time when the run record is not committed yet
(the retrain workflow promotes before it commits). ``--git-sha`` and ``--trained-at``
override both. The version string is ``<model>-<train_end_repdte>-<git_sha>``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
RUN_NAME = "walkforward"


def latest_year(walkforward_dir: Path) -> int:
    """The largest ``<Y>`` directory under ``models/walkforward/``; raises when there is none."""
    years = [int(p.name) for p in walkforward_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not years:
        raise FileNotFoundError(f"no walk-forward years under {walkforward_dir}")
    return max(years)


def run_record_dir(config: dict, runs_dir: Path) -> Path:
    """``runs/walkforward/<run_id>/`` for this model config (the walk-forward logs one per fit)."""
    from bankcanary.tracking import run_id

    return runs_dir / RUN_NAME / run_id(RUN_NAME, config)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text or None


def training_commit(run_dir: Path, artefact: Path, repo: Path = ROOT) -> tuple[str, str, str]:
    """``(git_sha, trained_at, source)`` of the fit that produced ``artefact``.

    The commit that added ``run_dir/metrics.json`` is the training commit and its committer
    date the training time (``source = 'run-record'``); when that record is untracked the
    short ``HEAD`` sha and the artefact's modification time are used (``source = 'head'``).
    """
    record = run_dir / "metrics.json"
    if record.exists():
        line = _git(["log", "--diff-filter=A", "--format=%h %cI", "--", str(record)], cwd=repo)
        if line:
            sha, when = line.splitlines()[-1].split(" ", 1)
            return sha, when, "run-record"
    sha = _git(["rev-parse", "--short", "HEAD"], cwd=repo) or "unknown"
    when = datetime.fromtimestamp(artefact.stat().st_mtime, tz=UTC).isoformat()
    return sha, when, "head"


def model_version(model: str, train_end_repdte: str, git_sha: str) -> str:
    """CONTRACT section 15: ``<model>-<train_end_repdte>-<git short sha>``."""
    return f"{model}-{train_end_repdte}-{git_sha}"


def promote_model(
    model: str,
    year: int,
    walkforward_dir: Path,
    production_dir: Path,
    runs_dir: Path,
    horizon: int = 4,
    git_sha: str | None = None,
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
    run_dir = run_record_dir(config, runs_dir)
    sha, when, source = training_commit(run_dir, src / "pipeline.joblib", repo)
    sha, when = git_sha or sha, trained_at or when
    if git_sha or trained_at:
        source = "override"
    dst = production_dir / model
    dst.mkdir(parents=True, exist_ok=True)
    for name in ARTEFACTS:
        shutil.copyfile(src / name, dst / name)
    record = {
        "model_version": model_version(model, config["train_repdte_max"], sha),
        "model": model,
        "train_end_repdte": config["train_repdte_max"],
        "git_sha": sha,
        "trained_at": when,
        "features_version": config["features_version"],
        "horizon": int(config["horizon"]),
        "walkforward_year": int(config["test_year"]),
        "source": str(src.relative_to(repo)) if src.is_relative_to(repo) else str(src),
        "provenance": source,
        "notes": (
            f"Walk-forward {model} for test year {config['test_year']}, trained on reports "
            f"{config['train_repdte_min']}..{config['train_repdte_max']} "
            f"({config['n_train']:,} rows, {config['positives_train']:,} failures); "
            "calibration.joblib is that year's isotonic map."
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
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--git-sha", default=None, help="override the training commit")
    parser.add_argument("--trained-at", default=None, help="override the training timestamp")
    args = parser.parse_args(argv)
    walkforward_dir = args.models_dir / "walkforward"
    year = args.year if args.year is not None else latest_year(walkforward_dir)
    for model in args.models:
        record = promote_model(
            model,
            year,
            walkforward_dir,
            args.models_dir / "production",
            args.runs_dir,
            args.horizon,
            args.git_sha,
            args.trained_at,
        )
        print(
            f"{record['model_version']}: {record['source']} -> models/production/{model} "
            f"(trained_at {record['trained_at']}, {record['provenance']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
