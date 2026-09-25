"""The three Prototype 1 baselines: Texas ratio, small logistic, regularised logistic.

* ``texas``: rank banks by the Texas ratio alone (non-performing assets over tangible
  capital plus reserves). No fitting; the classic supervisory rule of thumb and the bar
  every learned model has to clear.
* ``logit_small``: logistic regression on six CAMELS proxies (capital, asset quality,
  earnings, funding, concentration, size), Cole and White style, so the odds ratios can
  be read one by one.
* ``logit``: L2 logistic regression on every registered Prototype 1 feature, including
  the missing-value flags and the charter-class one-hots, with balanced class weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from bankcanary.features.registry import feature_names
from bankcanary.models.preprocess import make_pipeline

MODEL_NAMES: tuple[str, ...] = ("texas", "logit_small", "logit")

#: The six-feature specification of contract section 8.
SMALL_FEATURES: tuple[str, ...] = (
    "equity_to_assets",
    "noncurrent_ratio",
    "roa_q",
    "brokered_share",
    "construction_to_capital",
    "log_assets",
)

#: Score assigned to a missing Texas ratio: below every real (non-negative) value.
TEXAS_MISSING_SCORE = -1.0


class TexasRatioScorer(BaseEstimator):
    """Score = the ``texas_ratio`` column itself; a missing ratio ranks last.

    The ratio is already capped at 10 by the feature layer, so no winsorising or
    scaling is applied: clipping the top half-percent would tie exactly the banks a
    ranking metric cares about most.
    """

    def __init__(self, column: str = "texas_ratio"):
        self.column = column

    def fit(self, X, y=None):
        self._column_index(X)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def _column_index(self, X) -> int:
        if isinstance(X, pd.DataFrame):
            if self.column not in X.columns:
                raise KeyError(f"{self.column!r} not among the input columns")
            return int(X.columns.get_loc(self.column))
        if np.ndim(X) != 2 or X.shape[1] != 1:
            raise ValueError("array input must be a single column holding the Texas ratio")
        return 0

    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        j = self._column_index(X)
        values = np.asarray(X, dtype=float)[:, j]
        return np.where(np.isnan(values), TEXAS_MISSING_SCORE, values)

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) >= 1.0).astype(int)


def model_features(name: str) -> list[str]:
    """Feature columns a named baseline consumes, in registry order."""
    if name == "texas":
        return ["texas_ratio"]
    if name == "logit_small":
        return list(SMALL_FEATURES)
    if name == "logit":
        return feature_names("P1")
    raise ValueError(f"unknown model {name!r}; choose one of {MODEL_NAMES}")


def make_model(name: str, max_iter: int = 2000) -> Pipeline:
    """Build the unfitted pipeline for a baseline name.

    Both logistic models use L2 regularisation at ``C=1.0`` with ``class_weight="balanced"``
    (failures are about 0.3 percent of rows) and the shared winsorise-impute-scale front
    end. The Texas baseline is wrapped in a one-step pipeline so every model loads and
    scores the same way.
    """
    if name == "texas":
        return Pipeline([("model", TexasRatioScorer())])
    if name in ("logit_small", "logit"):
        estimator = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=max_iter, solver="lbfgs"
        )
        return make_pipeline(estimator)
    raise ValueError(f"unknown model {name!r}; choose one of {MODEL_NAMES}")


def score_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Failure score for ranking: ``predict_proba[:, 1]`` when available, else the raw score."""
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(X))[:, 1]
    return np.asarray(pipeline.decision_function(X), dtype=float)


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients of a fitted logistic pipeline on the standardised, winsorised inputs.

    Columns: ``feature`` (registry name, or ``missingindicator_<name>`` for the imputer
    flags), ``coef``, ``odds_ratio`` (= exp(coef), the multiplicative change in failure
    odds per one training-fold standard deviation) and ``abs_coef``; sorted by
    ``abs_coef`` descending.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise TypeError("coefficient_table needs a fitted linear model in the 'model' step")
    names = pipeline[:-1].get_feature_names_out()
    coef = np.asarray(model.coef_).ravel()
    table = pd.DataFrame({"feature": names, "coef": coef})
    table["odds_ratio"] = np.exp(table["coef"])
    table["abs_coef"] = table["coef"].abs()
    return table.sort_values("abs_coef", ascending=False, kind="mergesort").reset_index(drop=True)
