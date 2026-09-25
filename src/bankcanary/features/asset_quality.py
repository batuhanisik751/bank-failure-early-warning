"""Asset quality features (the A in CAMELS).

Most bank failures are credit failures: loans stop performing, reserves prove too thin
and the losses eat through capital. These ratios track that chain from early
delinquency, through noncurrent loans and charge-offs, to the Texas ratio that
compares problem assets with the capital left to absorb them.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.registry import RATIO_CAP, safe_ratio
from bankcanary.features.ytd import annualize, average_with_previous, deaccumulate


def texas_ratio(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Problem assets over tangible equity plus reserves, with the cap flag.

    When the denominator is zero or negative the bank has no tangible cushion at all,
    so the ratio is set to ``RATIO_CAP`` (10) and ``texas_ratio_capped`` is True. A
    computed ratio above the cap is also clipped and flagged so the column has a
    bounded, monotone meaning. Missing inputs keep NaN with the flag False.
    """
    intan = panel["intan"].astype("float64").fillna(0.0)
    problem = panel["nclnls"].astype("float64") + panel["ore"]
    cushion = panel["eq"].astype("float64") - intan + panel["lnatres"]
    ratio = problem / cushion.where(cushion > 0)
    no_cushion = (cushion <= 0) & problem.notna()
    capped = no_cushion | (ratio > RATIO_CAP)
    value = ratio.where(~capped, RATIO_CAP)
    return value, capped.fillna(False).astype(bool)


def reserve_coverage(panel: pd.DataFrame) -> pd.Series:
    """Allowance per dollar of noncurrent loans, capped at 10; zero noncurrent → 10."""
    reserves = panel["lnatres"].astype("float64")
    noncurrent = panel["nclnls"].astype("float64")
    ratio = safe_ratio(reserves, noncurrent).clip(upper=RATIO_CAP)
    return ratio.where(noncurrent != 0, RATIO_CAP).where(reserves.notna())


def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """Asset-quality features aligned to ``panel``'s index (see registry for definitions)."""
    out = pd.DataFrame(index=panel.index)
    out["noncurrent_ratio"] = safe_ratio(panel["nclnls"], panel["lnlsgr"])
    npa = panel["nclnls"].astype("float64") + panel["ore"]
    out["npa_to_assets"] = safe_ratio(npa, panel["asset"])
    out["early_delinquency"] = safe_ratio(panel["p3asset"], panel["lnlsgr"])
    quarterly = deaccumulate(panel, ["ntlnls"])["ntlnls_q"]
    out["nco_rate"] = safe_ratio(annualize(quarterly), average_with_previous(panel, "lnlsgr"))
    out["reserve_coverage"] = reserve_coverage(panel)
    out["texas_ratio"], out["texas_ratio_capped"] = texas_ratio(panel)
    return out
