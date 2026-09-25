"""Asset quality features (the A in CAMELS).

Most bank failures are credit failures: loans stop performing, reserves prove too thin
and the losses eat through capital. These ratios track that chain from early
delinquency, through noncurrent loans and charge-offs, to the Texas ratio that
compares problem assets with the capital left to absorb them.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import RATIO_CAP, FeatureSpec, safe_ratio, spec
from bankcanary.features.ytd import annualize, average_with_previous, deaccumulate

SPECS: list[FeatureSpec] = [
    spec(
        "noncurrent_ratio",
        "asset_quality",
        "nclnls / lnlsgr",
        "ratio",
        "Share of loans 90+ days past due or on nonaccrual; the most direct read on credit "
        "losses already in the pipeline.",
        monotone=1,
    ),
    spec(
        "npa_to_assets",
        "asset_quality",
        "(nclnls + ore) / asset",
        "ratio",
        "Non-performing assets, including foreclosed property, relative to the whole balance "
        "sheet.",
        monotone=1,
    ),
    spec(
        "early_delinquency",
        "asset_quality",
        "p3asset / lnlsgr",
        "ratio",
        "Assets 30-89 days past due over gross loans; an early signal before loans turn "
        "noncurrent.",
        monotone=1,
    ),
    spec(
        "nco_rate",
        "asset_quality",
        "annualised ntlnls_q / average lnlsgr",
        "ratio",
        "Annualised net charge-offs as a share of average loans; the losses actually realised "
        "this quarter.",
        monotone=1,
    ),
    spec(
        "reserve_coverage",
        "asset_quality",
        "lnatres / nclnls, capped at 10",
        "ratio",
        "Loan-loss allowance per dollar of noncurrent loans; low coverage means future losses "
        "will hit earnings and capital.",
        monotone=-1,
    ),
    spec(
        "texas_ratio",
        "asset_quality",
        "(nclnls + ore) / (eq - intan + lnatres), capped at 10",
        "ratio",
        "Problem assets over tangible equity plus reserves; values near or above 1 have "
        "historically preceded most failures.",
        monotone=1,
    ),
    spec(
        "texas_ratio_capped",
        "asset_quality",
        "texas_ratio hit the cap",
        "flag",
        "True when tangible capital plus reserves was zero or negative (or the ratio exceeded "
        "10), so the reported value is the cap rather than a computed ratio.",
        monotone=1,
    ),
]


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


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
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
