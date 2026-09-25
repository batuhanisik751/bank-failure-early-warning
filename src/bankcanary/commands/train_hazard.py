"""``bankcanary train-hazard``: the discrete-time hazard on the fixed split, converted to H q."""

from __future__ import annotations

import logging

import typer


def register(app: typer.Typer) -> None:
    @app.command("train-hazard")
    def train_hazard_cmd(
        horizon: int = typer.Option(
            4, "--horizon", help="Horizon in quarters the hazard is converted to and compared at."
        ),
        c: float | None = typer.Option(
            None, "--c", help="Inverse L2 strength; default bankcanary.models.hazard.HAZARD_C."
        ),
        tune: bool = typer.Option(
            False, "--tune", help="Only score the C grid on the inner validation slice and exit."
        ),
        skip_comparators: bool = typer.Option(
            False, "--skip-comparators", help="Do not score logit_v2 and gbdt at the horizon."
        ),
    ) -> None:
        """Fit the 1q hazard, save models/hazard, evaluate 1 - (1 - h)^H, write the report.

        The fit itself is the same in every call (rows come from the 1q fixed split);
        each call adds one converted horizon to ``models/hazard/metrics.json`` and
        re-renders ``reports/p2_hazard.md`` from everything saved so far. Comparators
        fitted on the horizon's own label are loaded from ``models/`` when present and
        fitted otherwise. ``--tune`` runs the C grid (one run record per candidate,
        already-scored candidates are read back) and prints the table.
        """
        from bankcanary.config import load_settings
        from bankcanary.models import gbdt, hazard

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = load_settings()
        frame = gbdt.load_training_frame(settings, hazard.FEATURE_VERSION)
        if tune:
            table = hazard.tune_c(frame, settings)
            typer.echo(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            best = table.iloc[0]
            typer.echo(
                f"winner: C={best['C']} (validation PR-AUC 4q {best['pr_auc_4q']:.4f}); "
                "copy it into bankcanary.models.hazard.HAZARD_C"
            )
            return
        chosen = hazard.HAZARD_C if c is None else float(c)
        result = hazard.fit_hazard(frame, settings, chosen)
        m = result.metrics
        typer.echo(
            f"hazard 1q test: pr_auc {m['pr_auc']:.4f} roc_auc {m['roc_auc']:.4f} "
            f"recall@2% {m['recall_at_2pct']:.4f} brier {m['brier']:.5f}"
        )
        from bankcanary.splits import fixed_split_masks

        train, _ = fixed_split_masks(frame, settings, hazard.HAZARD_HORIZON)
        odds = hazard.odds_ratio_table(frame, train, result.pipeline)
        hazard._write_json(
            hazard.hazard_dir(settings) / "odds_ratios.json",
            {"rows": odds.to_dict(orient="records")},
        )
        typer.echo(odds[["feature", "odds_ratio", "ci_low", "ci_high"]].round(3).to_string())
        comparators = () if skip_comparators else hazard.COMPARATORS
        block = hazard.evaluate_converted(
            result.pipeline, frame, settings, horizon, chosen, comparators
        )
        for name, mm in [("hazard", block["hazard"]), *block["comparators"].items()]:
            typer.echo(
                f"{name} {horizon}q: pr_auc {mm['pr_auc']:.4f} roc_auc {mm['roc_auc']:.4f} "
                f"recall@2% {mm['recall_at_2pct']:.4f} brier {mm['brier']:.5f}"
            )
        typer.echo(f"wrote {hazard.write_hazard_report(settings)}")
