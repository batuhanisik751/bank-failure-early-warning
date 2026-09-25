"""Loan concentration features (a supervisory lens across A and M in CAMELS).

Concentrated lending, above all in construction and commercial real estate, drove the
2008-2011 failure wave. Exposures are measured against total risk-based capital, the
denominator the 2006 interagency CRE guidance uses for its 100% / 300% thresholds.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.registry import safe_ratio

#: Loan-mix share features: name -> loan category column (all over ``lnlsgr``).
LOAN_SHARES = {
    "share_construction": "lnrecons",
    "share_nonfarm_nonres": "lnrenres",
    "share_multifamily": "lnremult",
    "share_residential": "lnreres",
    "share_ci": "lnci",
    "share_consumer": "lncon",
    "share_agri": "lnag",
}


def capital_base(panel: pd.DataFrame) -> pd.Series:
    """Total risk-based capital in dollars (``rbc``), else Tier 1 capital plus the allowance.

    Community-bank leverage-ratio filers (2020+) report ``rbc`` as zero, so those rows
    fall back to ``rbct1j + lnatres``, the conventional approximation of total capital.
    A base that is still not positive yields NaN ratios.
    """
    rbc = panel["rbc"].astype("float64")
    fallback = panel["rbct1j"].astype("float64") + panel["lnatres"]
    return rbc.where(rbc > 0, fallback)


def cre_exposure(panel: pd.DataFrame) -> pd.Series:
    """Construction + multifamily + non-owner-occupied nonfarm nonresidential loans.

    The owner-occupied split (``lnrenrot``) only exists from 2007; earlier rows use the
    total ``lnrenres``, which overstates CRE slightly but keeps the series defined.
    """
    nonres = panel["lnrenres"].astype("float64")
    if "lnrenrot" in panel.columns:
        non_owner = panel["lnrenrot"].astype("float64")
        nonres = non_owner.where(non_owner.notna(), nonres)
    return panel["lnrecons"].astype("float64") + panel["lnremult"] + nonres


def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """Concentration features aligned to ``panel``'s index (see registry for definitions)."""
    out = pd.DataFrame(index=panel.index)
    base = capital_base(panel)
    out["construction_to_capital"] = safe_ratio(panel["lnrecons"], base)
    out["cre_to_capital"] = safe_ratio(cre_exposure(panel), base)
    for name, col in LOAN_SHARES.items():
        out[name] = safe_ratio(panel[col], panel["lnlsgr"])
    return out
