"""Command-line entry point: ``bankcanary <command>``.

Commands are registered by the modules that implement them; this file only wires them up.
"""

from __future__ import annotations

import importlib.util
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
    force: bool = typer.Option(False, "--force", help="Re-download even when cached."),
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
        if importlib.util.find_spec("bankcanary.ingest.financials") is None:
            typer.echo("financials: not available yet (bankcanary.ingest.financials missing)")
        else:
            from bankcanary.ingest import financials

            entry = getattr(financials, "ingest_financials", None) or financials.fetch_financials
            entry(settings, force=force)


if __name__ == "__main__":
    app()
