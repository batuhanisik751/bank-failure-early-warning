"""``bankcanary publish``: write the warehouse and production scores to Postgres (CONTRACT 17)."""

from __future__ import annotations

import datetime as dt
import logging
import time

import typer


def register(app: typer.Typer) -> None:
    @app.command("publish")
    def publish_cmd(
        tables: str | None = typer.Option(
            None, "--tables", help="Comma-separated subset of the tables (default: all)."
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Build every frame and print row counts; no database."
        ),
        schema_only: bool = typer.Option(
            False, "--schema-only", help="Only apply schema.sql (tables and indexes)."
        ),
    ) -> None:
        """Rebuild the published tables from the warehouse and ``models/production/``.

        Reads ``DATABASE_URL`` from the environment or ``.env``. Each table is loaded
        inside one transaction (truncate + COPY); ``pipeline_runs`` keeps one row per
        run and ``runs/publish/`` a tracking record. Educational project, not a credit
        rating, not investment advice, not a supervisory assessment.
        """
        from bankcanary import tracking
        from bankcanary.config import load_settings
        from bankcanary.publish import (
            ALL_TABLES,
            CORE_TABLES,
            DISCLAIMER,
            PAGE_TABLES,
            core,
            db,
            pages,
            writer,
        )

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        wanted = list(ALL_TABLES)
        if tables:
            wanted = [t.strip() for t in tables.split(",") if t.strip()]
            unknown = sorted(set(wanted) - set(ALL_TABLES))
            if unknown:
                raise typer.BadParameter(f"unknown tables {unknown}; choose from {ALL_TABLES}")
        page_tables = [t for t in PAGE_TABLES if t in wanted]
        core_tables = [t for t in CORE_TABLES if t in wanted] + sorted(
            pages.core_dependencies(page_tables) - set(wanted)
        )
        started = dt.datetime.now(dt.UTC)
        clock = time.perf_counter()
        if schema_only:
            with db.connect() as conn:
                writer.apply_schema(conn)
            typer.echo("schema applied")
            return
        warehouse = core.load_warehouse(settings, set(wanted))
        frames = core.build_all(warehouse, settings, core_tables)
        frames.update(pages.build_all(warehouse, frames, settings, page_tables))
        frames = {t: frames[t] for t in ALL_TABLES if t in wanted}
        for name, frame in frames.items():
            typer.echo(f"{name}: {len(frame)} rows")
        if dry_run:
            typer.echo(f"dry run: nothing written ({time.perf_counter() - clock:.0f} s)")
            return
        latest = warehouse.panel["repdte"].max() if len(warehouse.panel) else None
        run = tracking.start_run(
            "publish",
            {"horizon": core.HORIZON, "tables": wanted, "started_at": started.isoformat()},
            settings,
        )
        written: dict[str, int] = {}
        with db.connect() as conn:
            writer.apply_schema(conn)
            for name, frame in frames.items():
                written[name] = writer.write_table(conn, name, frame, mode="replace")
            writer.analyze(conn, list(written))
            finished = dt.datetime.now(dt.UTC)
            row = core.pipeline_run_row(
                run.run_id,
                started,
                finished,
                "ok",
                written,
                f"published {len(written)} tables in {time.perf_counter() - clock:.0f} s",
                latest_repdte=None if latest is None else latest.date(),
            )
            writer.write_table(conn, "pipeline_runs", row, mode="upsert")
            size = db.database_size_bytes(conn)
        run.log_metrics({**{f"rows_{k}": v for k, v in written.items()}, "db_bytes": size})
        run.finish()
        typer.echo(f"database size: {size / 1e6:.1f} MB; run {run.run_id}")
        typer.echo(DISCLAIMER)
