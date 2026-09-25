"""Assemble the ``features_v1`` table from the panel (CONTRACT sections 5 and 7).

``build_features`` evaluates every CAMELS module and returns the registered features
in registry order, keyed by ``(cert, repdte)``, one row per panel row. Nothing is
winsorised or imputed here: that happens inside the model pipeline, fitted on the
training fold only, so no test-period information leaks into the features.
"""

from __future__ import annotations

import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.features import (
    asset_quality,
    capital,
    concentration,
    earnings,
    liquidity,
    structure,
)
from bankcanary.features.registry import feature_names
from bankcanary.features.ytd import KEY, YTD_PREV_MISSING, deaccumulate

log = logging.getLogger(__name__)

TABLE = "features_v1"
MODULES = (capital, asset_quality, earnings, liquidity, concentration, structure)

#: Features compared between failing and surviving banks in the build summary.
CONTRAST_FEATURES = [
    "texas_ratio",
    "noncurrent_ratio",
    "roa_q",
    "equity_to_assets",
    "brokered_share",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Every registered P1 feature for every panel row, plus ``ytd_prev_missing``.

    The de-accumulation flag is shared by all income items (it depends only on whether
    the previous quarter-end row exists), so it is computed once from ``netinc``.
    """
    frames = [module.compute(panel) for module in MODULES]
    features = pd.concat(frames, axis=1)
    names = feature_names()
    missing = [n for n in names if n not in features.columns]
    if missing:
        raise RuntimeError(f"registered features not computed: {missing}")
    flag = deaccumulate(panel, ["netinc"])[YTD_PREV_MISSING]
    keys = panel[KEY].copy()
    keys["repdte"] = pd.to_datetime(keys["repdte"]).astype("datetime64[ns]")
    out = pd.concat([keys, features[names], flag], axis=1)
    return out.reset_index(drop=True)


def failure_contrast(
    features: pd.DataFrame, labels: pd.DataFrame, horizon: int = 4
) -> pd.DataFrame:
    """Median of the contrast features for label-complete rows, failing vs surviving.

    Rows flagged ``dropped_failed_before_avail`` or not label-complete for ``horizon``
    are excluded, mirroring the training filter, so the comparison is honest.
    """
    y = f"y_{horizon}q"
    keep = labels[f"label_complete_{horizon}q"].fillna(False).astype(bool) & ~labels[
        "dropped_failed_before_avail"
    ].fillna(True).astype(bool)
    joined = features.merge(labels.loc[keep, KEY + [y]], on=KEY, how="inner")
    return joined.groupby(y)[CONTRAST_FEATURES].median().T


def build_features_table(settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Read ``panel``, build ``features_v1``, write Parquet + DuckDB and summarise."""
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import read_table, table_path, write_table

    panel = read_table("panel", settings=settings)
    features = build_features(panel)
    path = write_table(features, TABLE, settings=settings)
    rows = replace_table(TABLE, path, settings=settings)
    summary: dict = {"rows": len(features), "features": len(feature_names()), "duckdb_rows": rows}
    if table_path(settings, "labels").exists():
        labels = read_table("labels", settings=settings)
        summary["contrast"] = failure_contrast(features, labels)
    log.info("%s: %d rows, %d features -> %s", TABLE, len(features), summary["features"], path)
    return features, summary
