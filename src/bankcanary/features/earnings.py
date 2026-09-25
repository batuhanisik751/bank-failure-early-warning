"""Earnings features (the E in CAMELS).

Income items in the Call Report are year-to-date, so each is first de-accumulated to a
single quarter (``features.ytd``), annualised (x4) and divided by a two-point average
balance so a bank that doubled in size mid-quarter is not flattered or punished.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, safe_ratio, spec
from bankcanary.features.ytd import annualize, average_with_previous, deaccumulate

#: Year-to-date income columns this module de-accumulates.
INCOME_COLS = ["netinc", "intinc", "eintexp", "nim", "nonii", "nonix", "elnatr"]


SPECS: list[FeatureSpec] = [
    spec(
        "roa_q",
        "earnings",
        "annualised netinc_q / average asset",
        "ratio",
        "Annualised quarterly return on average assets; the core measure of profitability.",
        monotone=-1,
    ),
    spec(
        "nim_q",
        "earnings",
        "annualised (intinc_q - eintexp_q) / average ernast (else asset)",
        "ratio",
        "Annualised net interest margin over average earning assets; the spread the bank "
        "earns on its lending.",
        monotone=0,
    ),
    spec(
        "efficiency_ratio",
        "earnings",
        "nonix_q / (nim_q + nonii_q)",
        "ratio",
        "Noninterest expense per dollar of revenue; higher means a costlier operation.",
        monotone=1,
    ),
    spec(
        "provision_rate",
        "earnings",
        "annualised elnatr_q / average lnlsgr",
        "ratio",
        "Annualised loan-loss provisions as a share of average loans; what management expects "
        "to lose.",
        monotone=1,
    ),
]


def _earning_asset_base(panel: pd.DataFrame) -> pd.Series:
    """Average earning assets, falling back to average total assets where missing.

    ``ernast`` (total earning assets) is the conventional margin denominator; a row
    without it uses average total assets so the margin is still defined.
    """
    avg_assets = average_with_previous(panel, "asset")
    if "ernast" not in panel.columns:
        return avg_assets
    avg_earning = average_with_previous(panel, "ernast")
    return avg_earning.where(avg_earning.notna() & (avg_earning > 0), avg_assets)


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Earnings features aligned to ``panel``'s index (see registry for definitions).

    Note the name clash that is *not* a bug: the de-accumulated net interest income is
    the intermediate column ``nim_q`` (FDIC code ``NIM`` = net interest income YTD),
    while the returned feature ``nim_q`` is the annualised margin ratio.
    """
    q = deaccumulate(panel, INCOME_COLS)
    avg_assets = average_with_previous(panel, "asset")
    avg_loans = average_with_previous(panel, "lnlsgr")
    out = pd.DataFrame(index=panel.index)
    out["roa_q"] = safe_ratio(annualize(q["netinc_q"]), avg_assets)
    net_interest = q["intinc_q"] - q["eintexp_q"]
    out["nim_q"] = safe_ratio(annualize(net_interest), _earning_asset_base(panel))
    revenue = q["nim_q"] + q["nonii_q"]
    out["efficiency_ratio"] = safe_ratio(q["nonix_q"], revenue)
    out["provision_rate"] = safe_ratio(annualize(q["elnatr_q"]), avg_loans)
    return out
