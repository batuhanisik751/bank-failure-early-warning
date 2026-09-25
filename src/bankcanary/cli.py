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


if __name__ == "__main__":
    app()
