"""Pick the L2 strength of the all-feature logit on a validation slice inside the training period.

Spec rule 6.7: hyperparameters are tuned inside the training period, never on the test
years. The inner split mirrors the fixed one: validation = usable reports 2007Q1-2008Q4,
inner training = training rows whose outcome window closed before the validation
prediction date (2007-05-30, so reports up to 2005Q4). Every candidate is fitted on the
inner training rows and scored on the validation rows; the test split is never touched
here. Run ``uv run python scripts/tune_logit_c.py`` and copy the winner into
``bankcanary.models.baselines.LOGIT_C``.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression

from bankcanary.config import load_settings
from bankcanary.evaluation.metrics import evaluate
from bankcanary.models.baselines import model_features
from bankcanary.models.preprocess import make_pipeline
from bankcanary.models.train import load_training_frame
from bankcanary.splits import fixed_split_masks, test_mask, training_mask

VALIDATION_START = "2007-03-31"
VALIDATION_END = "2008-12-31"
GRID = (1.0, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    frame = load_training_frame(settings)
    horizon, lag = 4, settings.availability_lag_days
    outer_train, _ = fixed_split_masks(frame, settings, horizon)
    inner_train = training_mask(frame, horizon, VALIDATION_START, lag) & outer_train
    validation = test_mask(frame, horizon, VALIDATION_START, VALIDATION_END) & outer_train
    features, y_col = model_features("logit"), f"y_{horizon}q"
    print(
        f"inner train {int(inner_train.sum()):,} rows / {int(frame.loc[inner_train, y_col].sum())} "
        f"positives (last report {frame.loc[inner_train, 'repdte'].max().date()}); "
        f"validation {int(validation.sum()):,} rows / {int(frame.loc[validation, y_col].sum())} "
        "positives"
    )
    rows = []
    for weighting in (None, "balanced"):
        for c in GRID:
            pipe = make_pipeline(LogisticRegression(C=c, class_weight=weighting, max_iter=2000))
            pipe.fit(frame.loc[inner_train, features], frame.loc[inner_train, y_col].astype(int))
            m = evaluate(
                frame.loc[validation, y_col].astype(int).to_numpy(),
                pipe.predict_proba(frame.loc[validation, features])[:, 1],
                tie_breaker=frame.loc[validation, "cert"].to_numpy(),
            )
            rows.append({"class_weight": weighting or "none", "C": c, **m})
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    cols = ["class_weight", "C", "pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = table.iloc[0]
    print(f"\nwinner: class_weight={best['class_weight']} C={best['C']} (validation PR-AUC)")


if __name__ == "__main__":
    main()
