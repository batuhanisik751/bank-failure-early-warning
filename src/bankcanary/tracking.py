"""Lightweight run tracking: one JSON pair per run under ``runs/`` (CONTRACT section 11).

Every training, tuning, backtest, calibration or sensitivity call opens a run with
:func:`start_run`, logs its metrics and finishes. A run is identified by its *config*,
never by the clock: ``run_id = <name>-<horizon>q-<sha1 of the canonical config>[:10]``,
so re-running the same command overwrites the same small files instead of piling up
timestamped copies, and a tuning script can ask :func:`find_metrics` whether a
configuration has already been scored. ``runs/index.jsonl`` lists every run once
(deduplicated by id) with its scalar metrics so the history is greppable without opening
each directory; ``finish`` only ever appends to it and :func:`rebuild_index` rewrites it
from the run directories. The directory is committed; keep the metric payloads small.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
    """Every run in ``runs/index.jsonl`` once, deduplicated by id (empty before any run).

    The file may hold several lines for one id (each ``finish`` appends, see
    :func:`_append_index`); the latest line wins and keeps the position of the first.
    """
    path = runs_dir(settings) / INDEX_FILE
    if not path.exists():
        return []
    return _dedupe(_read_rows(path))


def rebuild_index(settings: Settings) -> int:
    """Rewrite ``runs/index.jsonl`` from every ``runs/<name>/<run_id>/config.json``.

    One row per run directory, sorted by run id, built from the files on disk rather than
    from the old index, so rows lost or duplicated by interleaved calls are repaired and
    two rebuilds of the same tree give byte-identical files. The new file replaces the old
    one atomically (temporary file + rename). Returns the number of rows written.
    """
    root = runs_dir(settings)
    rows = []
    for config_path in sorted(root.glob("*/*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics_path = config_path.with_name("metrics.json")
        metrics = {} if not metrics_path.exists() else json.loads(metrics_path.read_text())
        run_dir = config_path.parent
        run = Run(run_dir.parent.name, run_dir.name, config, run_dir, settings, metrics)
        rows.append(run.index_row())
    rows.sort(key=lambda r: r["run_id"])
    _replace_file(root / INDEX_FILE, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return len(rows)


#: Run names whose records live only while a saved model references them: the model's
#: own ``walkforward`` record and its ``tune_walkforward`` candidates
#: (:func:`bankcanary.evaluation.walkforward.referenced_run_ids`).
REFERENCED_RUN_NAMES: tuple[str, ...] = ("walkforward", "tune_walkforward")
#: Run names that keep one record per subject, the newest; an earlier generation of the
#: same analysis (a recalibration, a re-run explanation) is pruned.
DEDUPED_RUN_NAMES: tuple[str, ...] = (
    "sensitivity",
    "case_study_2023",
    "explain",
    "calibrate",
    "metrics",
)
#: Config keys that say what a deduplicated run is about. The rest of a config (fitted
#: hyper-parameters, slice dates, row and failure counts, wall-clock stamps) is derived
#: from one generation's data and would keep every generation alive.
SUBJECT_KEYS: tuple[str, ...] = (
    "model",
    "horizon",
    "fit_horizon",
    "test_year",
    "year",
    "model_year",
    "analysis",
    "variant",
    "view",
    "features_version",
    "method",
    "cut_repdte",
    "lead_time_years",
    "top_frac",
    "n_draws",
    "seed",
)


def run_subject(name: str, config: dict) -> str:
    """Canonical JSON of the :data:`SUBJECT_KEYS` present in ``config``, with the run name."""
    return canonical_json({"name": name, **{k: config[k] for k in SUBJECT_KEYS if k in config}})


def _record_mtime(run_dir: Path) -> float:
    files = (run_dir / "metrics.json", run_dir / "config.json")
    stamps = [p.stat().st_mtime for p in files if p.exists()]
    return max(stamps) if stamps else 0.0


def prune_plan(settings: Settings, referenced: set[str]) -> list[Path]:
    """The run directories :func:`prune` would delete, in path order.

    A :data:`REFERENCED_RUN_NAMES` record goes unless its id is in ``referenced``; a
    :data:`DEDUPED_RUN_NAMES` record goes when a newer record (by metrics/config mtime,
    then run id) shares its :func:`run_subject`. Every other name is left alone, and so
    is any directory without a ``config.json``.
    """
    root = runs_dir(settings)
    doomed: list[Path] = []
    for name_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        name = name_dir.name
        records = sorted(d for d in name_dir.iterdir() if (d / "config.json").exists())
        if name in REFERENCED_RUN_NAMES:
            doomed += [d for d in records if d.name not in referenced]
        elif name in DEDUPED_RUN_NAMES:
            newest: dict[str, tuple[tuple[float, str], Path]] = {}
            for d in records:
                config = json.loads((d / "config.json").read_text(encoding="utf-8"))
                subject, key = run_subject(name, config), (_record_mtime(d), d.name)
                if subject not in newest or key > newest[subject][0]:
                    newest[subject] = (key, d)
            keep = {d for _, d in newest.values()}
            doomed += [d for d in records if d not in keep]
    return doomed


def prune(settings: Settings, referenced: set[str], dry_run: bool = False) -> dict:
    """Delete the :func:`prune_plan` directories and rebuild the index (``dry_run``: plan only).

    Returns ``{"deleted": [paths relative to the runs root], "index_rows": n or None}``.
    """
    import shutil

    root = runs_dir(settings)
    plan = prune_plan(settings, referenced)
    if not dry_run:
        for run_dir in plan:
            shutil.rmtree(run_dir)
    rows = None if dry_run else rebuild_index(settings)
    return {"deleted": [str(d.relative_to(root)) for d in plan], "index_rows": rows}


def _append_index(path: Path, row: dict) -> None:
    """Append one line with a single ``O_APPEND`` write, so concurrent finishes never
    overwrite each other's rows; readers deduplicate by id (:func:`read_index`) and
    :func:`rebuild_index` compacts the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _replace_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _dedupe(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row.get("run_id")] = row  # a repeated id keeps its first slot, latest content
    return list(latest.values())


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
