"""Preprocessing shared by every estimator: winsorise, impute, scale (contract section 8).

Call Report ratios have heavy tails: a bank with a handful of loans can post an
efficiency ratio in the thousands, and a de-novo bank a 10x asset-growth rate. Those
values are real but would let a few rows steer a linear model, so the pipeline first
clips each feature to quantiles *estimated on the training fold only* (the test fold
is never consulted), then fills missing values with the training median while keeping a
"was missing" indicator, and finally standardises so coefficients are comparable.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip every column to its ``[lower, upper]`` training quantiles.

    Quantiles ignore missing values and missing values pass through unchanged, so the
    imputer that follows still sees them. Binary (0/1) columns such as one-hot charter
    classes and missing-value flags are left untouched when ``skip_binary`` is set: a
    rare flag (say 0.2 percent of rows) would otherwise be clipped to all zeros. A column
    that is entirely missing on the training fold is not clipped either.
    """

    def __init__(self, lower: float = 0.005, upper: float = 0.995, skip_binary: bool = True):
        self.lower = lower
        self.upper = upper
        self.skip_binary = skip_binary

    def fit(self, X, y=None):
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError(f"need 0 <= lower < upper <= 1, got {self.lower}, {self.upper}")
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite="allow-nan", reset=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column
            lo = np.nanquantile(X, self.lower, axis=0)
            hi = np.nanquantile(X, self.upper, axis=0)
        present = ~np.isnan(X)
        binary = np.array(
            [np.isin(X[present[:, j], j], (0.0, 1.0)).all() for j in range(X.shape[1])],
            dtype=bool,
        )
        skip = np.isnan(lo) | np.isnan(hi) | (binary if self.skip_binary else False)
        self.lower_bounds_ = np.where(skip, -np.inf, lo)
        self.upper_bounds_ = np.where(skip, np.inf, hi)
        self.skipped_ = skip
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite="allow-nan", reset=False)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def make_pipeline(estimator, lower: float = 0.005, upper: float = 0.995) -> Pipeline:
    """``Winsorizer -> SimpleImputer(median, add_indicator) -> StandardScaler -> estimator``."""
    return Pipeline(
        [
            ("winsorize", Winsorizer(lower=lower, upper=upper)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
