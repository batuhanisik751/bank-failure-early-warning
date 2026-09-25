"""``bankcanary train-gbdt``: gradient boosting on the fixed split, with the comparators."""

from __future__ import annotations

import logging

import typer

VARIANT_CHOICES = ("gbdt", "gbdt_mono", "both")


def register(app: typer.Typer) -> None:
    @app.command("train-gbdt")
    def train_gbdt_cmd(
        variant: str = typer.Option(
            "both", "--variant", help="gbdt (unconstrained), gbdt_mono (monotone) or both."
        ),
        horizon: int = typer.Option(4, "--horizon", help="Label horizon in quarters."),
        skip_comparators: bool = typer.Option(
            False, "--skip-comparators", help="Do not refit logit_v2 and score the Texas ratio."
        ),
    ) -> None:
        """Fit the tuned booster(s) on features_v2, save artefacts, log runs, write the report.

        Parameters and backend come from ``settings.models.gbdt`` (run
        ``scripts/tune_gbdt.py`` first). Besides the booster(s) the report carries the
        Texas ratio and the P1 logit refitted on the v2 features, so the gain of the
        trees over the linear model on the same inputs is visible.
        """
        from bankcanary.config import load_settings
        from bankcanary.models import gbdt
        from bankcanary.splits import fixed_split_masks

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        if variant not in VARIANT_CHOICES:
            raise typer.BadParameter(f"--variant must be one of {VARIANT_CHOICES}")
        settings = load_settings()
        if settings.models.gbdt.backend is None:
            raise typer.BadParameter(
                "settings.models.gbdt is not set; run 'uv run python scripts/tune_gbdt.py' first"
            )
        frame = gbdt.load_training_frame(settings, gbdt.FEATURE_VERSION)
        names = [] if skip_comparators else ["texas", "logit_v2"]
        names += ["gbdt", "gbdt_mono"] if variant == "both" else [variant]
        results, importance = {}, None
        for name in names:
            pipeline, features = gbdt.make_variant(name, settings)
            result = gbdt.fit_on_fixed_split(
                name, pipeline, features, frame, settings, horizon, save=name != "texas"
            )
            results[name] = result
            m = result.metrics
            typer.echo(
                f"{name}: test pr_auc {m['pr_auc']:.4f} roc_auc {m['roc_auc']:.4f} "
                f"recall@2% {m['recall_at_2pct']:.4f} recall@100 {m['recall_at_top100']:.4f}"
            )
            if name == "gbdt":
                # Permutation importance (sklearn backend only) reads a sample of test rows.
                _, test = fixed_split_masks(frame, settings, horizon)
                rows = frame.loc[test].sample(min(int(test.sum()), 20_000), random_state=0)
                importance = gbdt.feature_importance(
                    pipeline, features, rows[features], rows[f"y_{horizon}q"].to_numpy()
                )
        path = gbdt.write_gbdt_report(
            results, settings, horizon, gbdt.report_path(settings, horizon), importance
        )
        typer.echo(f"wrote {path}")
