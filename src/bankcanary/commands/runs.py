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

    @runs_app.command("prune")
    def prune_cmd(
        dry_run: bool = typer.Option(False, "--dry-run", help="Only list what would go."),
        runs_dir: str | None = typer.Option(None, "--runs-dir", help=RUNS_DIR_HELP),
        models_dir: str | None = typer.Option(
            None, "--models-dir", help="Models root (default: settings.models_dir)."
        ),
    ) -> None:
        """Delete records no saved model references and older generations of the analyses.

        ``walkforward`` and ``tune_walkforward`` records stay only while a
        ``models/walkforward/<Y>/<model>/config.json`` or
        ``models/production/<model>/config.json`` names them (its own record id and its
        tuning candidates); ``sensitivity``, ``case_study_2023``, ``explain``,
        ``calibrate`` and ``metrics`` keep the newest record per subject (model, horizon,
        year, analysis, variant, view). Other names are untouched. Rebuilds
        ``runs/index.jsonl`` afterwards; ``--dry-run`` only prints the plan.
        """
        from bankcanary import tracking
        from bankcanary.evaluation import walkforward as w

        settings = _settings(runs_dir)
        root = Path(models_dir) if models_dir else Path(settings.models_dir)
        referenced = w.referenced_run_ids(w.saved_model_configs(root))
        result = tracking.prune(settings, referenced, dry_run=dry_run)
        verb = "would delete" if dry_run else "deleted"
        for path in result["deleted"]:
            typer.echo(f"{verb} {path}")
        typer.echo(
            f"{len(result['deleted'])} run(s) {verb}; {len(referenced)} referenced run ids"
            + ("" if dry_run else f"; index rebuilt: {result['index_rows']} runs")
        )

    app.add_typer(runs_app, name="runs")
