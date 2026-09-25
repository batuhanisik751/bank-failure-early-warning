"""``bankcanary sensitivity``: horizon, censoring and availability-lag refits (spec step 13)."""

from __future__ import annotations

import logging

import typer

WHICH_CHOICES = ("horizon", "censored", "lag", "all")


def register(app: typer.Typer) -> None:
    @app.command("sensitivity")
    def sensitivity_cmd(
        which: str = typer.Option("all", "--which", help="horizon | censored | lag | all."),
        model: str = typer.Option(
            "all", "--model", help="logit | gbdt | all (comma lists allowed)."
        ),
        no_report: bool = typer.Option(
            False, "--no-report", help="Log the runs but do not rewrite reports/sensitivity.md."
        ),
    ) -> None:
        """Refit logit and gbdt on the fixed split under each changed assumption, log a run each.

        ``horizon`` compares the 4q and 8q label sets, ``censored`` drops the non-failure
        exits from both training and test, ``lag`` rebuilds the labels in memory for a
        45, 60 and 90 day availability lag and recomputes the rule 6.2 split at that lag.
        One analysis per call stays inside the two-minute budget; ``--which all`` runs
        the three in sequence. The report is rebuilt from ``runs/sensitivity/`` afterwards,
        so partial results are always visible.
        """
        from bankcanary.config import load_settings
        from bankcanary.evaluation import sensitivity as s

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        if which not in WHICH_CHOICES:
            raise typer.BadParameter(f"--which must be one of {WHICH_CHOICES}")
        names = list(s.MODELS) if model == "all" else [m.strip() for m in model.split(",")]
        unknown = [m for m in names if m not in s.MODELS]
        if unknown:
            raise typer.BadParameter(f"unknown model(s) {unknown}; choose from {s.MODELS}")
        settings = load_settings()
        frame = s.load_frame(settings)
        for variant in s.variants(which, settings, names):
            result = s.fit_variant(frame, settings, variant)
            m = result.metrics
            typer.echo(
                f"{variant.analysis} {variant.name} {variant.model}: n {m['n']} failures "
                f"{m['n_failures']} pr_auc {m['pr_auc']:.4f} roc_auc {m['roc_auc']:.4f} "
                f"recall@2% {m['recall_at_2pct']:.4f} (train rows {result.config['n_train']}, "
                f"last {result.config['train_repdte_max']})"
            )
        if not no_report:
            typer.echo(f"wrote {s.write_sensitivity_report(settings)}")
