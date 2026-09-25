"""Assemble the ``features_v1`` / ``features_v2`` tables from the panel (CONTRACT 5, 7, 12).

``build_features`` runs every feature module of the requested version in registry
order and returns the registered columns keyed by ``(cert, repdte)``, one row per panel
row. ``features_v2`` is ``features_v1`` plus the Prototype 2 columns, so the v1 layout
never changes. Nothing is winsorised or imputed here: that happens inside the model
pipeline, fitted on the training fold only, so no test-period information leaks into
the features.
"""

from __future__ import annotations

import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.features import registry
from bankcanary.features.ytd import KEY, YTD_PREV_MISSING, deaccumulate

log = logging.getLogger(__name__)

TABLE = "features_v1"
TABLE_KEY = ("cert", "repdte")
VERSIONS = ("v1", "v2")

#: Features compared between failing and surviving banks in the build summary.
CONTRAST_FEATURES = [
    "texas_ratio",
    "noncurrent_ratio",
    "roa_q",
    "equity_to_assets",
    "brokered_share",
]


def table_name(version: str = "v1") -> str:
    """``features_v1`` / ``features_v2``."""
    if version not in VERSIONS:
        raise ValueError(f"unknown feature version {version!r}; expected one of {VERSIONS}")
    return f"features_{version}"


def column_order(version: str = "v1") -> list[str]:
    """Feature columns of ``features_<version>`` in table order (keys excluded).

    The P1 columns come first, then ``ytd_prev_missing`` (the end of the v1 layout), then
    the P2 columns, so every v2 table starts with the v1 table's columns.
    """
    names = registry.feature_names(version=version)
    p1 = [n for n in names if n in registry.feature_names(version="v1")]
    p2 = [n for n in names if n not in p1]
    return p1 + [YTD_PREV_MISSING] + p2


def build_features(panel: pd.DataFrame, version: str = "v1", **deps: object) -> pd.DataFrame:
    """Every registered feature of ``version`` for every panel row, plus ``ytd_prev_missing``.

    Modules run in registry order; each receives the columns built so far as
    ``deps["features"]`` (plus any extra ``deps`` the caller passes, such as a macro
    table), so a later module can derive trends from earlier ratios. The de-accumulation
    flag is shared by all income items (it depends only on whether the previous
    quarter-end row exists), so it is computed once from ``netinc``.
    """
    table_name(version)
    built = pd.DataFrame(index=panel.index)
    for module in registry.modules(version):
        frame = module.build(panel, features=built, **deps)
        expected = [s.name for s in module.SPECS]
        missing = [n for n in expected if n not in frame.columns]
        if missing:
            raise RuntimeError(f"{module.__name__} did not compute {missing}")
        built = pd.concat([built, frame[expected]], axis=1)
    flag = deaccumulate(panel, ["netinc"])[YTD_PREV_MISSING]
    keys = panel[KEY].copy()
    keys["repdte"] = pd.to_datetime(keys["repdte"]).astype("datetime64[ns]")
    out = pd.concat([keys, built, flag], axis=1)[KEY + column_order(version)]
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


def build_features_table(
    settings: Settings, version: str = "v1", deps: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Read ``panel``, build ``features_<version>``, write Parquet + DuckDB and summarise."""
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import read_table, table_path, write_table

    name = table_name(version)
    panel = read_table("panel", settings=settings)
    features = build_features(panel, version=version, **(deps or {}))
    path = write_table(features, name, key=TABLE_KEY, settings=settings)
    rows = replace_table(name, path, settings=settings)
    n_features = len(registry.feature_names(version=version))
    summary: dict = {"rows": len(features), "features": n_features, "duckdb_rows": rows}
    if table_path(settings, "labels").exists():
        labels = read_table("labels", settings=settings)
        summary["contrast"] = failure_contrast(features, labels)
    log.info("%s: %d rows, %d features -> %s", name, len(features), n_features, path)
    return features, summary
