"""Lightweight run tracking: one JSON pair per run under ``runs/`` (CONTRACT section 11).

Every training, tuning, backtest, calibration or sensitivity call opens a run with
:func:`start_run`, logs its metrics and finishes. A run is identified by its *config*,
never by the clock: ``run_id = <name>-<horizon>q-<sha1 of the canonical config>[:10]``,
so re-running the same command overwrites the same small files instead of piling up
timestamped copies, and a tuning script can ask :func:`find_metrics` whether a
configuration has already been scored. ``runs/index.jsonl`` lists every run once
(deduplicated by id) with its scalar metrics so the history is greppable without opening
each directory. The directory is committed; keep the metric payloads small.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bankcanary.config import Settings

log = logging.getLogger(__name__)

INDEX_FILE = "index.jsonl"


def canonical_json(config: dict) -> str:
    """The config as sorted, compact JSON: the text that is hashed into the run id."""
    return json.dumps(_json_ready(config), sort_keys=True, separators=(",", ":"))


def run_id(name: str, config: dict) -> str:
    """``<name>-<horizon>q-<sha1(canonical config)[:10]>``; ``config['horizon']`` is required."""
    if "horizon" not in config:
        raise KeyError("run config needs a 'horizon' entry")
    digest = hashlib.sha1(canonical_json(config).encode("utf-8")).hexdigest()[:10]
    return f"{name}-{int(config['horizon'])}q-{digest}"


def runs_dir(settings: Settings) -> Path:
    return Path(settings.runs_dir)


@dataclass
class Run:
    """One open run: ``log_metrics`` accumulates, ``finish`` writes the files and the index."""

    name: str
    run_id: str
    config: dict
    dir: Path
    settings: Settings = field(repr=False)
    metrics: dict = field(default_factory=dict)
    finished: bool = False

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(_json_ready(metrics))

    def finish(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        _dump(self.dir / "config.json", _json_ready(self.config))
        _dump(self.dir / "metrics.json", self.metrics)
        _append_index(runs_dir(self.settings) / INDEX_FILE, self.index_row())
        self.finished = True
        log.info("run %s finished (%s)", self.run_id, self.dir)
        return self.dir

    def index_row(self) -> dict:
        scalars = {k: v for k, v in self.metrics.items() if isinstance(v, int | float | str)}
        return {
            "run_id": self.run_id,
            "name": self.name,
            "horizon": int(self.config["horizon"]),
            "metrics": scalars,
        }


def start_run(name: str, config: dict, settings: Settings) -> Run:
    """Open a run named ``name`` for ``config``; its files land in ``runs/<name>/<run_id>/``."""
    rid = run_id(name, config)
    return Run(name, rid, dict(config), runs_dir(settings) / name / rid, settings)


def find_metrics(name: str, config: dict, settings: Settings) -> dict | None:
    """Metrics of an earlier finished run with this exact config, or ``None``."""
    path = runs_dir(settings) / name / run_id(name, config) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_index(settings: Settings) -> list[dict]:
    """Every row of ``runs/index.jsonl`` (empty list when no run was logged yet)."""
    path = runs_dir(settings) / INDEX_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _append_index(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [] if not path.exists() else _read_rows(path)
    rows = [r for r in rows if r.get("run_id") != row["run_id"]] + [row]
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def _read_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value):
    """Plain JSON types only: numpy scalars unwrapped, NaN -> null, paths -> str."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value
