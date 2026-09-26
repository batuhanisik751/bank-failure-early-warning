"""``bankcanary runs``: read and repair the run index kept by :mod:`bankcanary.tracking`."""

from __future__ import annotations

from pathlib import Path

import typer

#: The first of these scalar metrics present in a run's record is its headline figure.
HEADLINE_METRICS: tuple[str, ...] = (
    "pr_auc",
    "brier_calibrated",
    "brier",
    "brier_raw",
    "top_feature",
)

RUNS_DIR_HELP = "Runs root (default: settings.runs_dir)."

runs_app = typer.Typer(name="runs", help="List and rebuild the run index (runs/index.jsonl).")


def headline(metrics: dict) -> str:
    for key in HEADLINE_METRICS:
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
            return f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
    return "-"


def format_rows(rows: list[dict]) -> list[str]:
    """One aligned line per run: id, name, horizon, headline metric."""
    if not rows:
        return ["no runs logged"]
    id_w = max(len(r["run_id"]) for r in rows)
    name_w = max(len(r["name"]) for r in rows)
    return [
        f"{r['run_id']:<{id_w}}  {r['name']:<{name_w}}  {r['horizon']}q  {headline(r['metrics'])}"
        for r in rows
    ]


def _settings(runs_dir: str | None):
    """The project settings, with ``runs_dir`` replaced when the option is given."""
    from bankcanary.config import load_settings

    settings = load_settings()
    if runs_dir is None:
        return settings
    return settings.model_copy(update={"runs_dir": Path(runs_dir)})


def register(app: typer.Typer) -> None:
    @runs_app.command("list")
    def list_cmd(
        name: str | None = typer.Option(None, "--name", help="Only runs of this name."),
        limit: int = typer.Option(50, "--limit", help="Show at most the last N rows."),
        runs_dir: str | None = typer.Option(None, "--runs-dir", help=RUNS_DIR_HELP),
    ) -> None:
        """Print run id, name, horizon and headline metric from ``runs/index.jsonl``."""
        from bankcanary import tracking

        rows = tracking.read_index(_settings(runs_dir))
        if name is not None:
            rows = [r for r in rows if r.get("name") == name]
        rows = rows[-max(int(limit), 0) :] if limit else rows
        for line in format_rows(rows):
            typer.echo(line)

    @runs_app.command("rebuild-index")
    def rebuild_cmd(
        runs_dir: str | None = typer.Option(None, "--runs-dir", help=RUNS_DIR_HELP),
    ) -> None:
        """Rewrite ``runs/index.jsonl`` from every run directory (sorted by run id)."""
        from bankcanary import tracking

        typer.echo(f"index rebuilt: {tracking.rebuild_index(_settings(runs_dir))} runs")

    app.add_typer(runs_app, name="runs")
