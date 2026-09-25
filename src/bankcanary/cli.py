"""Command-line entry point: ``bankcanary <command>``.

Commands are registered by the modules that implement them; this file only wires them up.
"""

from __future__ import annotations

import logging

import typer

from bankcanary import __version__

app = typer.Typer(
    name="bankcanary",
    help="Early-warning system for US bank failures (educational project, not advice).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """BankCanary pipeline commands. Run a command with --help for its options."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


STRUCTURE_TABLES = ("failures", "institutions", "history")


@app.command()
def ingest(
    what: str = typer.Option(
        "all", "--what", help="failures | institutions | history | financials | all"
    ),
    start: str | None = typer.Option(
        None, "--start", help="financials: first quarter end (YYYY-MM-DD), default settings"
    ),
    end: str | None = typer.Option(
        None, "--end", help="financials: last quarter end (YYYY-MM-DD), default latest published"
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even when cached."),
    build: bool = typer.Option(
        True, "--build/--no-build", help="financials: rebuild the table after the pull"
    ),
) -> None:
    """Pull FDIC tables into Parquet (data/parquet) and DuckDB (data/warehouse.duckdb)."""
    from bankcanary.config import load_settings
    from bankcanary.ingest import failures, history, institutions
    from bankcanary.sources.fdic import FdicClient
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import write_table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    choices = ("failures", "institutions", "history", "financials", "all")
    if what not in choices:
        raise typer.BadParameter(f"--what must be one of {choices}")
    settings = load_settings()
    wanted = list(STRUCTURE_TABLES) + ["financials"] if what == "all" else [what]
    fetchers = {
        "failures": failures.fetch_failures,
        "institutions": institutions.fetch_institutions,
        "history": history.fetch_history,
    }
    with FdicClient(settings) as client:
        for name in [w for w in wanted if w in fetchers]:
            df = fetchers[name](settings, client=client, force=force)
            path = write_table(df, name, settings=settings)
            rows = replace_table(name, path, settings=settings)
            typer.echo(f"{name}: {len(df)} rows -> {path} and duckdb ({rows} rows)")
            if name == "failures":
                typer.echo("FAILURE events per year (2000 onward), FDIC failed-bank list:")
                typer.echo(failures.failures_per_year(df).to_string())
                failures.check_benchmarks(df)
                typer.echo(f"benchmark years {failures.BENCHMARK_YEARS} match")
    if "financials" in wanted:
        import datetime as dt

        from bankcanary.ingest import financials

        df = financials.ingest_financials(
            settings,
            start=dt.date.fromisoformat(start) if start else None,
            end=dt.date.fromisoformat(end) if end else None,
            force=force,
            build=build,
        )
        if df is not None:
            counts = financials.quarter_counts(df)
            typer.echo(f"financials_raw: {len(df)} rows, {len(counts)} quarters")
            typer.echo(counts.to_string())


@app.command("build-panel")
def build_panel_cmd() -> None:
    """Join financials_raw with institution attributes and exit facts into the panel table."""
    from bankcanary.config import load_settings
    from bankcanary.panel.build import build_panel

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _, s = build_panel(load_settings())
    typer.echo(f"panel: {s['rows']} rows, {s['banks']} banks (duckdb {s['duckdb_rows']} rows)")
    typer.echo(f"banks with no institutions record: {s['banks_missing_institution']}")
    typer.echo(
        f"failed banks 2001 onward: {s['failed_banks']}, matched to a prior report: "
        f"{s['failed_matched']} ({s['failed_matched_share']:.1%}, target >= 95%)"
    )
    typer.echo("exit_reason counts (one row per bank):")
    for reason, n in s["exit_reasons"].items():
        typer.echo(f"  {reason}: {n}")


@app.command("build-labels")
def build_labels_cmd() -> None:
    """Label every panel row per horizon: failure inside (avail_date, window_end]."""
    from bankcanary.config import load_settings
    from bankcanary.labels.build import build_labels_table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    _, s = build_labels_table(settings)
    typer.echo(
        f"labels: {s['rows']} rows (duckdb {s['duckdb_rows']} rows), as_of {s['as_of_date']}, "
        f"dropped_failed_before_avail: {s['dropped']}"
    )
    typer.echo("positives per report year among label-complete, non-dropped rows:")
    typer.echo(s["per_year"].to_string())
    for h in settings.horizons_quarters:
        typer.echo(
            f"y_{h}q: {s[f'pos_{h}q']} positives / {s[f'rows_{h}q']} usable rows "
            f"= {s[f'rate_{h}q']:.4%}"
        )


@app.command("build-features")
def build_features_cmd() -> None:
    """Compute the Prototype 1 CAMELS feature set from the panel into features_v1."""
    from bankcanary.config import load_settings
    from bankcanary.features.build import build_features_table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _, s = build_features_table(load_settings())
    typer.echo(
        f"features_v1: {s['rows']} rows, {s['features']} features (duckdb {s['duckdb_rows']} rows)"
    )
    if "contrast" in s:
        typer.echo("median feature values, label-complete rows, y_4q = 0 vs 1:")
        typer.echo(s["contrast"].round(4).to_string())


@app.command("dq-report")
def dq_report_cmd() -> None:
    """Write reports/data_quality.md and fill first_available in config/fields.yaml."""
    from bankcanary.config import load_settings
    from bankcanary.quality.report import build_report

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path, s = build_report(load_settings())
    m = s["matching"]
    typer.echo(f"report: {path}")
    typer.echo(
        f"financials_raw: {s['rows']} rows, {s['quarters']} quarters, latest {s['latest_repdte']}"
    )
    typer.echo(f"first_available updated for {s['fields_updated']} field(s) in config/fields.yaml")
    typer.echo(
        f"failures 2001 onward: {m['failures']}, matched to a prior report: {m['matched']} "
        f"({m['share']:.1%})"
    )
    typer.echo(f"structural missingness changes: {len(s['structural_changes'])}")


@app.command()
def train(
    model: str = typer.Option(..., "--model", help="texas | logit_small | logit | all"),
    horizon: int = typer.Option(4, "--horizon", help="Label horizon in quarters (4 or 8)."),
) -> None:
    """Fit a baseline on the fixed out-of-time split and save it under models/<name>/."""
    from bankcanary.config import load_settings
    from bankcanary.models.baselines import MODEL_NAMES
    from bankcanary.models.train import load_training_frame, train_model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if model != "all" and model not in MODEL_NAMES:
        raise typer.BadParameter(f"--model must be one of {MODEL_NAMES + ('all',)}")
    settings = load_settings()
    frame = load_training_frame(settings)
    for name in MODEL_NAMES if model == "all" else [model]:
        r = train_model(name, horizon, settings, frame=frame)
        c = r.config
        typer.echo(
            f"{name} {horizon}q: train {c['train_repdte_min']}..{c['train_repdte_max']} "
            f"({c['n_train']} rows, {c['positives_train']} positives), test "
            f"{c['test_repdte_min']}..{c['test_repdte_max']} ({c['n_test']} rows, "
            f"{c['positives_test']} positives)"
        )
        typer.echo(
            f"  pr_auc {r.metrics['pr_auc']:.4f}  roc_auc {r.metrics['roc_auc']:.4f}  "
            f"recall@2% {r.metrics['recall_at_2pct']:.3f}  "
            f"recall@top100 {r.metrics['recall_at_top100']:.3f} -> {r.paths['dir']}"
        )


@app.command()
def evaluate(
    model: str | None = typer.Option(None, "--model", help="One model; default: all trained."),
    horizon: int = typer.Option(4, "--horizon", help="Label horizon in quarters (4 or 8)."),
) -> None:
    """Re-score saved baselines on the fixed test split; write reports/p1_baselines.md + figures."""
    from bankcanary.config import load_settings
    from bankcanary.models.train import (
        evaluate_model,
        load_training_frame,
        report_path,
        write_baselines_report,
        write_figures,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    frame = load_training_frame(settings)
    results = evaluate_model(model, horizon, settings, frame=frame)
    if not results:
        typer.echo("no trained model found; run `bankcanary train --model all` first")
        raise typer.Exit(code=1)
    for name, r in results.items():
        m = r.metrics
        typer.echo(
            f"{name} {horizon}q: pr_auc {m['pr_auc']:.4f} roc_auc {m['roc_auc']:.4f} "
            f"recall@2% {m['recall_at_2pct']:.3f} recall@top100 {m['recall_at_top100']:.3f} "
            f"(n {m['n']}, failures {m['n_failures']})"
        )
        typer.echo(
            f"  censored dropped: pr_auc {r.sensitivity['pr_auc']:.4f} "
            f"(n_dropped {r.sensitivity['n_dropped']}); per event: pr_auc "
            f"{r.by_event['pr_auc']:.4f} (events {r.by_event['n_events']}, "
            f"multi-bank {r.by_event['n_multi_bank_events']})"
        )
    figures = write_figures(results, settings, horizon)
    path = write_baselines_report(
        results, settings, horizon, report_path(settings, horizon), figures=figures
    )
    typer.echo(f"report: {path}")
    typer.echo(f"figures: {len(figures)} under {figures['recall_at_k'].parent}")


from bankcanary.commands import register_all  # noqa: E402

register_all(app)


if __name__ == "__main__":
    app()
