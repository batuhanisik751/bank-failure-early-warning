"""``bankcanary build-crosswalk``: the RSSD crosswalk table and the FFIEC cross-check report."""

from __future__ import annotations

import logging
from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    @app.command("build-crosswalk")
    def build_crosswalk(
        crosscheck: bool = typer.Option(
            True,
            "--crosscheck/--no-crosscheck",
            help="Also write reports/ffiec_crosscheck.md from financials_raw.",
        ),
        ffiec_zip: str | None = typer.Option(
            None,
            "--ffiec-zip",
            help="Bulk 'Call Reports -- Single Period' ZIP to read RC-B / RC-O values from.",
        ),
        download: bool = typer.Option(
            False, "--download", help="Download the 2022-12-31 bulk ZIP if it is not cached."
        ),
    ) -> None:
        """Build crosswalk_rssd (cert -> fed_rssd) from institutions; parquet + DuckDB."""
        from bankcanary.config import load_settings
        from bankcanary.ingest import crosswalk as cw
        from bankcanary.storage.duckdb import replace_table
        from bankcanary.storage.parquet import read_table, write_table

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        institutions = read_table("institutions", settings=settings)
        table = cw.build_crosswalk(institutions)
        missing = cw.missing_rssd_count(table)
        path = write_table(table, cw.TABLE, key=cw.KEY, settings=settings)
        rows = replace_table(cw.TABLE, path, settings=settings)
        typer.echo(f"{cw.TABLE}: {len(table)} certs, {missing} without an RSSD id -> {path}")
        typer.echo(f"duckdb {cw.TABLE}: {rows} rows")
        if not crosscheck:
            return
        financials = read_table("financials_raw", settings=settings)
        check = cw.securities_crosscheck(financials)
        cw.check_published(check)
        typer.echo("Silicon Valley Bank 2022-12-31 FDIC fields match its 10-K")
        zip_path = Path(ffiec_zip) if ffiec_zip else None
        if zip_path is None:
            from bankcanary.sources import ffiec

            candidate = ffiec.zip_path("2022-12-31", ffiec.cache_dir(settings))
            if candidate.exists():
                zip_path = candidate
            elif download:
                zip_path = ffiec.download_single_period("2022-12-31", settings=settings)
        if zip_path is not None:
            check = cw.add_ffiec_values(check, zip_path, table)
        report = cw.render_report(
            check, len(table), missing, zip_path.name if zip_path is not None else None
        )
        out = settings.reports_dir / "ffiec_crosscheck.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        typer.echo(f"wrote {out}")
