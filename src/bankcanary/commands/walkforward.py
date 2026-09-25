"""``bankcanary walkforward`` and ``walkforward-report``: the per-year backtest (spec 8.2)."""

from __future__ import annotations

import logging

import typer


def register(app: typer.Typer) -> None:
    @app.command("walkforward")
    def walkforward_cmd(
        year: int = typer.Option(..., "--year", help="Test year Y (2008 .. latest complete)."),
        model: str = typer.Option(
            "all", "--model", help="texas | logit | gbdt | hazard | all (comma lists allowed)."
        ),
        horizon: int = typer.Option(4, "--horizon", help="Scoring horizon in quarters."),
        gbdt_iterations: int | None = typer.Option(
            None, "--gbdt-iterations", help="Cap the booster's iteration count for this year."
        ),
        no_rebuild: bool = typer.Option(
            False, "--no-rebuild", help="Skip rebuilding the walkforward_scores table."
        ),
    ) -> None:
        """Fit and score every requested model for one test year, then rebuild the table.

        One call handles one year so that no command exceeds the two-minute budget.
        Artefacts land in ``models/walkforward/<Y>/<model>/``, scores in
        ``data/walkforward/<Y>_<model>_<H>q.parquet``; the ``walkforward_scores`` Parquet
        and DuckDB copies are rebuilt from every per-year file afterwards. Each fit logs a
        ``runs/walkforward/`` record keyed by its config.
        """
        from bankcanary.config import load_settings
        from bankcanary.evaluation import walkforward as w
        from bankcanary.models import gbdt

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        names = list(w.MODELS) if model == "all" else [m.strip() for m in model.split(",")]
        unknown = [m for m in names if m not in w.MODELS]
        if unknown:
            raise typer.BadParameter(f"unknown model(s) {unknown}; choose from {w.MODELS}")
        frame = gbdt.load_training_frame(settings, w.FEATURE_VERSION)
        years = w.test_years(frame, horizon)
        if year not in years:
            raise typer.BadParameter(f"{year} is outside the {horizon}q test years {years}")
        for name in names:
            cap = gbdt_iterations if name == "gbdt" else None
            result = w.fit_year(frame, settings, year, name, horizon, n_estimators=cap)
            m = result.metrics
            typer.echo(
                f"{year} {name} {horizon}q: n {m['n']} failures {m['n_failures']} "
                f"pr_auc {w._round(m['pr_auc'])} roc_auc {w._round(m['roc_auc'])} "
                f"recall@2% {w._round(m['recall_at_2pct'])} "
                f"(train rows {result.config['n_train']}, last {result.config['train_repdte_max']})"
            )
        if not no_rebuild:
            table = w.rebuild_scores_table(settings)
            typer.echo(f"walkforward_scores: {len(table)} rows")

    @app.command("walkforward-report")
    def walkforward_report_cmd(
        horizon: int = typer.Option(4, "--horizon", help="Horizon given the full treatment."),
    ) -> None:
        """Rebuild ``walkforward_scores`` from the per-year files, write reports/walkforward.md."""
        from bankcanary.config import load_settings
        from bankcanary.evaluation import walkforward as w

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        w.rebuild_scores_table(settings)
        table = w.read_scores_table(settings, horizon)
        pooled = w.pooled_table(w.per_year_tables(table))
        typer.echo(pooled.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        typer.echo(f"wrote {w.write_walkforward_report(settings, horizon)}")
