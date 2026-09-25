"""``bankcanary build-macro``: pull FRED series in chunks and build ``macro_state``."""

from __future__ import annotations

import logging

import typer


def register(app: typer.Typer) -> None:
    @app.command("build-macro")
    def build_macro_cmd(
        pull_only: bool = typer.Option(
            False, "--pull-only", help="Only fill the FRED cache; do not build the table."
        ),
        chunk: int = typer.Option(
            30, "--chunk", help="Series to download in this call (keeps each call short)."
        ),
        force: bool = typer.Option(False, "--force", help="Re-download the chunk even if cached."),
    ) -> None:
        """Pull state and national FRED series, then evaluate them point-in-time per panel row.

        Series already in ``data/raw/fred/`` are skipped, so the command is re-run until
        every series is cached; the table is only built once nothing is missing.
        """
        from bankcanary.config import load_secrets, load_settings
        from bankcanary.ingest import macro
        from bankcanary.sources.fred import FredClient

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        todo = macro.series_ids() if force else macro.missing_series(settings)
        batch = todo[: max(int(chunk), 0)]
        if batch:
            with FredClient(settings, secrets=load_secrets()) as client:
                typer.echo(f"pulling {len(batch)} FRED series via {client.source} ...")
                for sid in batch:
                    frame = client.series(sid, force=force)
                    typer.echo(f"  {sid}: {len(frame)} observations")
        remaining = macro.missing_series(settings)
        if remaining:
            typer.echo(f"{len(remaining)} series still missing; re-run build-macro to continue")
            raise typer.Exit(code=0)
        if pull_only:
            typer.echo("FRED cache complete (--pull-only)")
            raise typer.Exit(code=0)
        df, s = macro.build_macro(settings)
        typer.echo(
            f"macro_state: {s['rows']} rows, {s['states']} states, "
            f"{s['series_cached']}/{s['series_expected']} series (duckdb {s['duckdb_rows']} rows)"
        )
        typer.echo("share of rows with a value:")
        for col, share in s["coverage"].items():
            typer.echo(f"  {col}: {share:.1%}")
