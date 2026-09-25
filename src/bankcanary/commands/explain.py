"""``bankcanary explain``: SHAP drivers per bank-quarter (spec 7, 9; step D10)."""

from __future__ import annotations

import logging

import typer


def register(app: typer.Typer) -> None:
    @app.command("explain")
    def explain_cmd(
        year: int | None = typer.Option(None, "--year", help="Explain one walk-forward test year."),
        latest: bool = typer.Option(
            False,
            "--latest",
            help="Explain the quarters past the backtest with the newest booster.",
        ),
        all_years: bool = typer.Option(
            False, "--all", help="Every test year, then the latest rows."
        ),
        report: bool = typer.Option(
            False, "--report", help="Only rebuild the table and write reports/shap_summary.md."
        ),
        no_rebuild: bool = typer.Option(False, "--no-rebuild", help="Skip rebuilding the table."),
    ) -> None:
        """Run ``shap.TreeExplainer`` over a walk-forward booster and store the top drivers.

        ``--year Y`` explains Y's test rows with Y's own ``gbdt``; ``--latest`` explains
        every report quarter after the last complete test year with the most recent
        booster (the production model); ``--all`` loops over both (about 10 s a year,
        so a full loop takes about three minutes: batch years with ``--year`` when a
        call must stay under two minutes).
        Each call writes ``data/drivers/<Y>_gbdt.parquet``, logs a ``runs/explain/``
        record holding the mean |SHAP| per feature, and rebuilds the ``drivers`` table
        (Parquet + DuckDB). ``--report`` writes ``reports/shap_summary.md`` and the
        beeswarm figure from the stored records.
        """
        from bankcanary.config import load_settings
        from bankcanary.evaluation import walkforward as w
        from bankcanary.explain import shap_drivers as sd
        from bankcanary.models import gbdt

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        targets: list[int | str] = []
        if year is not None:
            targets.append(int(year))
        if latest:
            targets.append(sd.PRODUCTION)
        if not targets and not all_years and not report:
            raise typer.BadParameter("give --year Y, --latest, --all or --report")
        frame = None
        if targets or all_years or report:
            frame = gbdt.load_training_frame(settings, w.FEATURE_VERSION)
        if all_years:
            targets = [*w.test_years(frame, sd.HORIZON), sd.PRODUCTION]
        for target in targets:
            drivers, mean_abs = sd.explain_year(frame, settings, target)
            head = ", ".join(f"{f} {v:.3f}" for f, v in mean_abs.head(3).items())
            typer.echo(f"{target}: {len(drivers)} driver rows; top mean |SHAP|: {head}")
        if targets and not no_rebuild:
            table = sd.rebuild_drivers_table(settings)
            typer.echo(f"drivers: {len(table)} rows")
        if report:
            if not no_rebuild:
                sd.rebuild_drivers_table(settings)
            path = sd.write_summary(frame, settings)
            typer.echo(f"wrote {path}")
