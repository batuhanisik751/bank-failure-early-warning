"""Tune the gradient booster on the validation slice inside the training period (rule 6.7).

The split is the one ``scripts/tune_logit_c.py`` uses: validation = usable reports
2007Q1-2008Q4, inner training = fixed-split training rows whose outcome window closed
before the validation prediction date (2007-05-30, so reports up to 2005Q4). Every grid
point is fitted on the inner rows and scored on the validation rows, for both the
unconstrained and the monotone variant; the test years are never touched.

Each scored configuration is logged through ``bankcanary.tracking`` under
``runs/tune_gbdt/``, so a re-run skips what is already scored. One call fits at most
``--chunk`` configurations and stops after ``--budget`` seconds; run the script again
until it reports the grid complete, at which point it writes the winner (parameters,
backend, and the monotone decision = the better variant on inner PR-AUC) into
``config/settings.yaml`` under ``models.gbdt``.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time

import pandas as pd

from bankcanary import tracking
from bankcanary.config import DEFAULT_SETTINGS_PATH, load_settings
from bankcanary.evaluation.metrics import evaluate
from bankcanary.models.baselines import LOGIT_C
from bankcanary.models.gbdt import (
    FEATURE_VERSION,
    gbdt_features,
    load_training_frame,
    make_gbdt,
    resolve_backend,
    write_gbdt_settings,
)
from bankcanary.splits import assert_no_leakage, fixed_split_masks, test_mask, training_mask

VALIDATION_START = "2007-03-31"
VALIDATION_END = "2008-12-31"
RUN_NAME = "tune_gbdt"
HORIZON = 4
GRID: dict[str, tuple] = {
    "learning_rate": (0.03, 0.1),
    "num_leaves": (15, 31, 63),
    "min_samples_leaf": (50, 200),
    "n_estimators": (200, 400),
}
SPLIT = {"validation_start": VALIDATION_START, "validation_end": VALIDATION_END}


def grid_configs(backend: str) -> list[dict]:
    """Every (parameters x monotone) point as a tracking config; the grid order is fixed."""
    configs = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), values))
        for monotone in (False, True):
            configs.append(_config("gbdt_mono" if monotone else "gbdt", backend, params))
    return configs


def _config(model: str, backend: str | None, params: dict) -> dict:
    return {
        "model": model,
        "horizon": HORIZON,
        "features_version": FEATURE_VERSION,
        "backend": backend,
        "monotone": model == "gbdt_mono",
        "params": params,
        "split": SPLIT,
    }


def inner_masks(frame: pd.DataFrame, settings) -> tuple[pd.Series, pd.Series]:
    """``(inner_train, validation)`` exactly as in ``tune_logit_c.py``, leakage-checked."""
    lag = settings.availability_lag_days
    outer_train, _ = fixed_split_masks(frame, settings, HORIZON)
    inner_train = training_mask(frame, HORIZON, VALIDATION_START, lag) & outer_train
    validation = test_mask(frame, HORIZON, VALIDATION_START, VALIDATION_END) & outer_train
    assert_no_leakage(frame.loc[inner_train], HORIZON, VALIDATION_START, lag)
    return inner_train, validation


def score_config(config: dict, frame, inner_train, validation, features) -> dict:
    """Fit one configuration on the inner rows and return its validation metrics."""
    y_col = f"y_{HORIZON}q"
    if config["model"] == "logit_v2":
        from sklearn.linear_model import LogisticRegression

        from bankcanary.models.preprocess import make_pipeline

        pipe = make_pipeline(LogisticRegression(C=LOGIT_C, max_iter=2000))
    else:
        pipe = make_gbdt(config["backend"], config["monotone"], features, **config["params"])
    started = time.perf_counter()
    pipe.fit(frame.loc[inner_train, features], frame.loc[inner_train, y_col].astype(int))
    metrics = evaluate(
        frame.loc[validation, y_col].astype(int).to_numpy(),
        pipe.predict_proba(frame.loc[validation, features])[:, 1],
        tie_breaker=frame.loc[validation, "cert"].to_numpy(),
    )
    metrics["fit_seconds"] = round(time.perf_counter() - started, 2)
    return metrics


def main(chunk: int, budget: float) -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    backend = resolve_backend(settings.models.gbdt.backend)
    frame = load_training_frame(settings, FEATURE_VERSION)
    inner_train, validation = inner_masks(frame, settings)
    features, y_col = gbdt_features(), f"y_{HORIZON}q"
    print(
        f"backend {backend}; inner train {int(inner_train.sum()):,} rows / "
        f"{int(frame.loc[inner_train, y_col].sum())} positives (last report "
        f"{frame.loc[inner_train, 'repdte'].max().date()}); validation "
        f"{int(validation.sum()):,} rows / {int(frame.loc[validation, y_col].sum())} positives"
    )
    configs = grid_configs(backend) + [_config("logit_v2", None, {"C": LOGIT_C})]
    todo = [c for c in configs if tracking.find_metrics(RUN_NAME, c, settings) is None]
    print(f"{len(configs) - len(todo)} of {len(configs)} configurations already scored")
    started = time.perf_counter()
    for config in todo[: max(chunk, 0)]:
        if time.perf_counter() - started > budget:
            break
        metrics = score_config(config, frame, inner_train, validation, features)
        run = tracking.start_run(RUN_NAME, config, settings)
        run.log_metrics(metrics)
        run.finish()
        print(
            f"  {config['model']:<9} {config['params']} pr_auc {metrics['pr_auc']:.4f} "
            f"({metrics['fit_seconds']}s)"
        )
    remaining = [c for c in configs if tracking.find_metrics(RUN_NAME, c, settings) is None]
    if remaining:
        print(f"{len(remaining)} configurations still to score; re-run the script")
        return
    finish(configs, settings, backend)


def results_table(configs: list[dict], settings) -> pd.DataFrame:
    """One row per scored configuration with its cached validation metrics."""
    rows = []
    for config in configs:
        metrics = tracking.find_metrics(RUN_NAME, config, settings) or {}
        rows.append({"model": config["model"], **config["params"], **metrics})
    return pd.DataFrame(rows)


def finish(configs: list[dict], settings, backend: str) -> None:
    """Print the full grid, pick the best inner PR-AUC and write ``models.gbdt`` to settings."""
    table = results_table(configs, settings)
    cols = ["model", *GRID.keys(), "pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100"]
    cols += ["fit_seconds"]
    gb = table[table["model"] != "logit_v2"].sort_values("pr_auc", ascending=False)
    print(gb[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = gb.iloc[0]
    params = {k: (float(best[k]) if k == "learning_rate" else int(best[k])) for k in GRID}
    same = gb[(gb[list(GRID)] == pd.Series(params)).all(axis=1)].set_index("model")
    inner = {m: float(same.loc[m, "pr_auc"]) for m in ("gbdt", "gbdt_mono")}
    inner["logit_v2"] = float(table.loc[table["model"] == "logit_v2", "pr_auc"].iloc[0])
    monotone = best["model"] == "gbdt_mono"
    print(
        f"\nwinner: {best['model']} {params} (inner PR-AUC {best['pr_auc']:.4f}; other variant "
        f"at the same parameters {min(inner['gbdt'], inner['gbdt_mono']):.4f}; logit_v2 "
        f"{inner['logit_v2']:.4f}); slowest fit {gb['fit_seconds'].max():.1f}s"
    )
    path = write_gbdt_settings(DEFAULT_SETTINGS_PATH, backend, monotone, params, inner)
    print(f"wrote models.gbdt (backend={backend}, monotone={monotone}) to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk", type=int, default=16, help="configurations per call")
    parser.add_argument("--budget", type=float, default=100.0, help="seconds before stopping")
    args = parser.parse_args()
    main(args.chunk, args.budget)
