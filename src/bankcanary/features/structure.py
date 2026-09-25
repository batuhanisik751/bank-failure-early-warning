"""Structure features (size, age, charter) and the growth helper used by ``management``.

Management quality is not observable in the Call Report, so growth is used as its
proxy: banks that expand far faster than peers tend to be loosening underwriting.
Growth is a log ratio against the value exactly ``n`` quarter-ends earlier for the same
cert (``features.ytd.lag``), so filing gaps give NaN rather than a stale comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bankcanary.features.spec import BKCLASS_CODES, BKCLASS_NAMES, FeatureSpec, log_ratio, spec
from bankcanary.features.ytd import lag

DAYS_PER_YEAR = 365.25


SPECS: list[FeatureSpec] = [
    spec(
        "log_assets",
        "structure",
        "log(asset)",
        "log_thousands_usd",
        "Natural log of total assets in thousands of dollars; bank size on a scale where a "
        "doubling is a constant step.",
        monotone=-1,
    ),
    spec(
        "bank_age_years",
        "structure",
        "(repdte - estymd) / 365.25 days",
        "years",
        "Years since the charter was established; young (de novo) banks fail more often.",
        monotone=-1,
    ),
    spec(
        "has_holding_company",
        "structure",
        "rssdhcr present",
        "flag",
        "True when the bank sits under a holding company that can inject capital.",
        monotone=-1,
    ),
]

SPECS.extend(
    spec(
        f"bkclass_{code}",
        "structure",
        f"bkclass == '{code}'",
        "flag",
        f"True when the charter class is {code}, that is {BKCLASS_NAMES[code]}.",
        monotone=0,
    )
    for code in BKCLASS_CODES
)


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


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Structure + management features aligned to ``panel``'s index."""
    out = pd.DataFrame(index=panel.index)
    asset = panel["asset"].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_assets"] = np.log(asset.where(asset > 0).to_numpy())
    age_days = (pd.to_datetime(panel["repdte"]) - pd.to_datetime(panel["estymd"])).dt.days
    out["bank_age_years"] = age_days.astype("float64") / DAYS_PER_YEAR
    out["has_holding_company"] = panel["has_holding_company"].fillna(False).astype(bool)
    return pd.concat([out, one_hot_bkclass(panel["bkclass"])], axis=1)
