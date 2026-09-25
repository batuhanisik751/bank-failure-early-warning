"""Structure and growth features (size, age, charter, and the management growth proxies).

Management quality is not observable in the Call Report, so growth is used as its
proxy: banks that expand far faster than peers tend to be loosening underwriting.
Growth is a log ratio against the value exactly ``n`` quarter-ends earlier for the same
cert (``features.ytd.lag``), so filing gaps give NaN rather than a stale comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bankcanary.features.registry import BKCLASS_CODES, log_ratio
from bankcanary.features.ytd import lag

DAYS_PER_YEAR = 365.25


def growth(panel: pd.DataFrame) -> pd.DataFrame:
    """Management proxies: log growth of assets (4q, 12q) and gross loans (4q)."""
    out = pd.DataFrame(index=panel.index)
    out["asset_growth_4q"] = log_ratio(panel["asset"], lag(panel, "asset", 4))
    out["asset_growth_12q"] = log_ratio(panel["asset"], lag(panel, "asset", 12))
    out["loan_growth_4q"] = log_ratio(panel["lnlsgr"], lag(panel, "lnlsgr", 4))
    return out


def one_hot_bkclass(bkclass: pd.Series) -> pd.DataFrame:
    """``bkclass_<code>`` boolean columns for every registered charter class.

    A code outside :data:`BKCLASS_CODES` (or a missing value) leaves every column False.
    """
    codes = bkclass.astype("string").str.strip().str.upper()
    out = pd.DataFrame(index=bkclass.index)
    for code in BKCLASS_CODES:
        out[f"bkclass_{code}"] = (codes == code).fillna(False).astype(bool).to_numpy()
    return out


def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """Structure + management features aligned to ``panel``'s index."""
    out = growth(panel)
    asset = panel["asset"].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_assets"] = np.log(asset.where(asset > 0).to_numpy())
    age_days = (pd.to_datetime(panel["repdte"]) - pd.to_datetime(panel["estymd"])).dt.days
    out["bank_age_years"] = age_days.astype("float64") / DAYS_PER_YEAR
    out["has_holding_company"] = panel["has_holding_company"].fillna(False).astype(bool)
    return pd.concat([out, one_hot_bkclass(panel["bkclass"])], axis=1)
