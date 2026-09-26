"""Back-fill content-addressed ``model_version`` strings into saved model configs.

``uv run python scripts/stamp_model_versions.py [--models-dir models]`` loads every
``models/walkforward/<Y>/<model>/pipeline.joblib`` and
``models/production/<model>/pipeline.joblib``,
hashes the fitted estimator (:func:`bankcanary.evaluation.walkforward.stamp_model_version`)
and rewrites ``config.json`` with ``model_hash`` and ``model_version`` (CONTRACT section 15).
A production ``model_version.json`` beside a stamped config gets the same ``model_version``
and ``model_hash``. Nothing is refitted, scored or read from git; a config already carrying
the right version is left byte-identical. Prints one line per directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from bankcanary.evaluation.walkforward import stamp_model_version

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = "model_version.json"


def model_dirs(models_dir: Path) -> list[Path]:
    """Every walk-forward and production model directory holding a pipeline and a config."""
    found = list((models_dir / "walkforward").glob("*/*"))
    found += list((models_dir / "production").glob("*"))
    return sorted(
        d for d in found if (d / "pipeline.joblib").exists() and (d / "config.json").exists()
    )


def stamp_dir(directory: Path) -> tuple[str, bool]:
    """Stamp one directory; returns ``(model_version, changed)``."""
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    before = (config.get("model_version"), config.get("model_hash"))
    stamp_model_version(config, joblib.load(directory / "pipeline.joblib"))
    changed = before != (config["model_version"], config["model_hash"])
    if changed:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    version_path = directory / VERSION_FILE
    if version_path.exists():
        record = json.loads(version_path.read_text(encoding="utf-8"))
        if record.get("model_version") != config["model_version"] or "model_hash" not in record:
            record["model_version"] = config["model_version"]
            record["model_hash"] = config["model_hash"]
            record.pop("git_sha", None)
            version_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            changed = True
    return config["model_version"], changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args(argv)
    dirs = model_dirs(args.models_dir)
    if not dirs:
        print(f"no model directories under {args.models_dir}")
        return 1
    n_changed = 0
    for directory in dirs:
        version, changed = stamp_dir(directory)
        n_changed += changed
        rel = directory.relative_to(args.models_dir)
        print(f"{rel}: {version}{'' if changed else ' (unchanged)'}")
    print(f"{len(dirs)} directories, {n_changed} stamped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
