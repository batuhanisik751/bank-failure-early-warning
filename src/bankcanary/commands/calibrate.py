"""``bankcanary calibrate`` and ``metrics-report``: isotonic calibration and the metrics suite."""

from __future__ import annotations

import logging

import typer


def register(app: typer.Typer) -> None:
    @app.command("calibrate")
    def calibrate_cmd(
        year: int | None = typer.Option(None, "--year", help="Test year Y to calibrate."),
        all_years: bool = typer.Option(False, "--all-years", help="Every test year in turn."),
        model: str = typer.Option(
            "all", "--model", help="logit | gbdt | hazard | all (comma lists allowed)."
        ),
        horizon: int = typer.Option(4, "--horizon", help="Scoring horizon in quarters."),
        no_rebuild: bool = typer.Option(
            False, "--no-rebuild", help="Skip rebuilding the walkforward_scores table."
        ),
    ) -> None:
        """Fit the isotonic map for one walk-forward year and fill ``score_calibrated``.

        For each probability model the calibration slice is the last complete label
        year inside the year's training window; an inner model with the year's tuned
        configuration (from ``models/walkforward/<Y>/<model>/config.json``) scores it and
        ``IsotonicRegression(out_of_bounds="clip")`` is fitted there, then applied to the
        year's test scores. One year per call keeps each run short; ``--all-years`` loops
        over the years for an idle machine. Writes ``calibration.joblib`` next to the model,
        fills ``score_calibrated`` in ``data/walkforward/<Y>_<model>_<H>q.parquet``, rebuilds
        the ``walkforward_scores`` table and logs a ``runs/calibrate/`` record per fit.
        """
        from bankcanary.config import load_settings
        from bankcanary.evaluation import calibration as c
        from bankcanary.evaluation import walkforward as w
        from bankcanary.models import gbdt

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        if (year is None) == (not all_years):
            raise typer.BadParameter("give exactly one of --year Y or --all-years")
        settings = load_settings()
        names = list(c.MODELS) if model == "all" else [m.strip() for m in model.split(",")]
        unknown = [m for m in names if m not in c.MODELS]
        if unknown:
            raise typer.BadParameter(f"unknown model(s) {unknown}; choose from {c.MODELS}")
        frame = gbdt.load_training_frame(settings, w.FEATURE_VERSION)
        years = w.test_years(frame, horizon)
        if year is not None and year not in years:
            raise typer.BadParameter(f"{year} is outside the {horizon}q test years {years}")
        for y in years if all_years else [year]:
            for r in c.calibrate_year(frame, settings, y, names, horizon, rebuild=False):
                m = r.metrics
                typer.echo(
                    f"{y} {r.model} {horizon}q: brier raw {m['brier_raw']:.5f} -> calibrated "
                    f"{m['brier_calibrated']:.5f} (failure rate {m['failure_rate']:.5f}, mean "
                    f"calibrated {m['mean_calibrated']:.5f}); slice "
                    f"{r.config['calibration_start']}..{r.config['calibration_end']} with "
                    f"{r.config['positives_calibration']} failures, inner fit through "
                    f"{r.config['inner_train_repdte_max']}"
                    + ("" if r.config["sufficient"] else " [thin slice]")
                )
        if not no_rebuild:
            table = w.rebuild_scores_table(settings)
            typer.echo(f"walkforward_scores: {len(table)} rows")

    @app.command("metrics-report")
    def metrics_report_cmd(
        horizon: int = typer.Option(4, "--horizon", help="Horizon given the full treatment."),
    ) -> None:
        """Regenerate ``reports/walkforward.md`` with confidence intervals, Brier and lead time.

        Reads ``walkforward_scores`` (raw and calibrated scores) and the panel's failure
        dates; no model is refitted. Writes the per-year and pooled tables with 200-draw
        cluster-bootstrap intervals, the Brier scores raw and calibrated, the reliability
        deciles and the lead-time summary, plus ``reports/figures/reliability_<model>.png``
        and ``lead_time_<model>.png``, and logs a ``runs/metrics/`` record per model.
        """
        from bankcanary.config import load_settings
        from bankcanary.evaluation import metrics_report as r

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        typer.echo(f"wrote {r.write_metrics_report(settings, horizon)}")
