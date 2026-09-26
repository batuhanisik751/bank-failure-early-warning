"""``bankcanary refresh``: the scheduled pipeline (CONTRACT 17; spec E2)."""

from __future__ import annotations

import datetime as dt
import logging

import typer


def register(app: typer.Typer) -> None:
    @app.command("refresh")
    def refresh_cmd(
        force_quarter: str | None = typer.Option(
            None,
            "--force-quarter",
            help="Treat this quarter end (YYYY-MM-DD) as new and republish it.",
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Probe and print the decision; ingest and write nothing."
        ),
    ) -> None:
        """Probe the FDIC API for a new quarter and, when there is one, publish it.

        Compares the latest ``REPDTE`` with ``quarters.max(repdte)`` in Postgres (the
        warehouse when the database is unreachable, which the run log says). A new
        quarter is ingested, the panel, labels and features_v2 are rebuilt from the
        raw cache, the quarter is scored with ``models/production/``, explained with
        SHAP and upserted; every run writes a ``pipeline_runs`` row and a
        ``runs/refresh/`` record. Reads ``DATABASE_URL`` from the environment or
        ``.env``. Educational project, not a credit rating, not investment advice,
        not a supervisory assessment.
        """
        from bankcanary import refresh
        from bankcanary.config import load_settings
        from bankcanary.publish import DISCLAIMER

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        forced = None
        if force_quarter:
            try:
                forced = dt.date.fromisoformat(force_quarter)
            except ValueError as exc:
                raise typer.BadParameter("--force-quarter must be YYYY-MM-DD") from exc
        result = refresh.run(load_settings(), force_quarter=forced, dry_run=dry_run)
        d = result.decision
        typer.echo(
            f"latest REPDTE (API): {d.latest_api}; published ({d.source}): {d.latest_published}"
        )
        typer.echo(f"decision: {'new quarter ' + str(d.quarter) if d.new_quarter else 'no-op'}")
        typer.echo(f"reason: {d.reason}")
        for name, rows in result.rows_written.items():
            typer.echo(f"  {name}: {rows} rows")
        typer.echo(f"status: {result.status}; steps: {result.timings.text() or 'none'}")
        if result.run_id:
            typer.echo(f"run {result.run_id}")
        if result.db_bytes is not None:
            typer.echo(f"database size: {result.db_bytes / 1e6:.1f} MB")
        typer.echo(DISCLAIMER)
