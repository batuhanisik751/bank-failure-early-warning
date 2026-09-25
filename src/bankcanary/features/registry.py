"""Feature registry for the Prototype 1 CAMELS feature set (CONTRACT section 7).

Every engineered feature is registered exactly once with its CAMELS group, formula,
unit and a plain-English banking explanation. ``build.build_features`` evaluates the
registry in order, so the column order of ``features_v1`` is the order below, and the
model code takes its feature list from :func:`feature_names` rather than from the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Charter classes present in the panel; ``bkclass_<code>`` one-hot columns.
#: N = national (OCC), NM = state non-member, SM = state Fed member, SB = savings bank,
#: SI = stock savings institution, SL = savings and loan, NC = non-insured commercial,
#: OI = other insured institution.
BKCLASS_CODES: tuple[str, ...] = ("N", "NM", "SM", "SB", "SI", "SL", "NC", "OI")

#: Ratios that would be infinite are capped here (Texas ratio, reserve coverage).
RATIO_CAP = 10.0


@dataclass(frozen=True)
class FeatureSpec:
    """One registered feature: what it is, how it is computed and why it matters."""

    name: str
    camels_group: str
    formula: str
    unit: str
    explanation: str
    prototype: str = "P1"


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series; a zero, negative or missing denominator gives NaN.

    Balance-sheet ratios are meaningless when the denominator is not a positive amount
    (a bank with no loans has no delinquency *rate*), so those rows are left missing for
    the imputer rather than filled with an infinite or sign-flipped value.
    """
    num = numerator.astype("float64")
    den = denominator.astype("float64")
    return (num / den.where(den > 0)).rename(None)


def log_ratio(current: pd.Series, previous: pd.Series) -> pd.Series:
    """``log(current / previous)``; NaN unless both values are positive."""
    cur = current.astype("float64").where(current > 0)
    prev = previous.astype("float64").where(previous > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.log(cur.to_numpy() / prev.to_numpy()), index=current.index)


def _f(name: str, group: str, formula: str, unit: str, explanation: str) -> FeatureSpec:
    return FeatureSpec(name, group, formula, unit, explanation, "P1")


REGISTRY: list[FeatureSpec] = [
    _f(
        "equity_to_assets",
        "capital",
        "eq / asset",
        "ratio",
        "Share of the balance sheet funded by shareholders; the cushion that absorbs losses "
        "before depositors are hit.",
    ),
    _f(
        "tier1_leverage",
        "capital",
        "rbc1aaj",
        "percent",
        "Regulatory Tier 1 leverage ratio; below 4% a bank is undercapitalised and below 2% "
        "it is critically undercapitalised.",
    ),
    _f(
        "total_rbc_ratio",
        "capital",
        "rbcrwaj",
        "percent",
        "Total risk-based capital over risk-weighted assets; missing for banks that elect "
        "the community bank leverage ratio from 2020.",
    ),
    _f(
        "total_rbc_ratio_missing",
        "capital",
        "rbcrwaj is missing",
        "flag",
        "True when the risk-based ratio is not reported, so the model can tell a CBLR "
        "filer from a genuinely low ratio.",
    ),
    _f(
        "tangible_equity_to_assets",
        "capital",
        "(eq - intan) / (asset - intan)",
        "ratio",
        "Equity after stripping goodwill and other intangibles, which are worthless in a "
        "resolution.",
    ),
    _f(
        "noncurrent_ratio",
        "asset_quality",
        "nclnls / lnlsgr",
        "ratio",
        "Share of loans 90+ days past due or on nonaccrual; the most direct read on credit "
        "losses already in the pipeline.",
    ),
    _f(
        "npa_to_assets",
        "asset_quality",
        "(nclnls + ore) / asset",
        "ratio",
        "Non-performing assets, including foreclosed property, relative to the whole "
        "balance sheet.",
    ),
    _f(
        "early_delinquency",
        "asset_quality",
        "p3asset / lnlsgr",
        "ratio",
        "Assets 30-89 days past due over gross loans; an early signal before loans turn "
        "noncurrent.",
    ),
    _f(
        "nco_rate",
        "asset_quality",
        "annualised ntlnls_q / average lnlsgr",
        "ratio",
        "Annualised net charge-offs as a share of average loans; the losses actually "
        "realised this quarter.",
    ),
    _f(
        "reserve_coverage",
        "asset_quality",
        "lnatres / nclnls, capped at 10",
        "ratio",
        "Loan-loss allowance per dollar of noncurrent loans; low coverage means future "
        "losses will hit earnings and capital.",
    ),
    _f(
        "texas_ratio",
        "asset_quality",
        "(nclnls + ore) / (eq - intan + lnatres), capped at 10",
        "ratio",
        "Problem assets over tangible equity plus reserves; values near or above 1 have "
        "historically preceded most failures.",
    ),
    _f(
        "texas_ratio_capped",
        "asset_quality",
        "texas_ratio hit the cap",
        "flag",
        "True when tangible capital plus reserves was zero or negative (or the ratio "
        "exceeded 10), so the reported value is the cap rather than a computed ratio.",
    ),
    _f(
        "asset_growth_4q",
        "management",
        "log(asset / asset 4 quarters earlier)",
        "log_ratio",
        "One-year balance-sheet growth; rapid growth is a classic management red flag.",
    ),
    _f(
        "asset_growth_12q",
        "management",
        "log(asset / asset 12 quarters earlier)",
        "log_ratio",
        "Three-year balance-sheet growth, capturing a sustained expansion strategy.",
    ),
    _f(
        "loan_growth_4q",
        "management",
        "log(lnlsgr / lnlsgr 4 quarters earlier)",
        "log_ratio",
        "One-year loan-book growth; fast lending growth often means loosened underwriting.",
    ),
    _f(
        "roa_q",
        "earnings",
        "annualised netinc_q / average asset",
        "ratio",
        "Annualised quarterly return on average assets; the core measure of profitability.",
    ),
    _f(
        "nim_q",
        "earnings",
        "annualised (intinc_q - eintexp_q) / average ernast (else asset)",
        "ratio",
        "Annualised net interest margin over average earning assets; the spread the bank "
        "earns on its lending.",
    ),
    _f(
        "efficiency_ratio",
        "earnings",
        "nonix_q / (nim_q + nonii_q)",
        "ratio",
        "Noninterest expense per dollar of revenue; higher means a costlier operation.",
    ),
    _f(
        "provision_rate",
        "earnings",
        "annualised elnatr_q / average lnlsgr",
        "ratio",
        "Annualised loan-loss provisions as a share of average loans; what management "
        "expects to lose.",
    ),
    _f(
        "brokered_share",
        "liquidity",
        "bro / dep",
        "ratio",
        "Share of deposits bought through brokers; hot money that leaves first when a bank "
        "weakens.",
    ),
    _f(
        "loans_to_deposits",
        "liquidity",
        "lnlsnet / dep",
        "ratio",
        "How far lending is funded by deposits; above 1 the bank relies on borrowed funds.",
    ),
    _f(
        "liquid_assets_ratio",
        "liquidity",
        "(chbal + frepo + sc) / asset",
        "ratio",
        "Cash, fed funds sold and securities as a share of assets; what can be sold or "
        "pledged quickly.",
    ),
    _f(
        "wholesale_funding_ratio",
        "liquidity",
        "(othbor + bro) / asset",
        "ratio",
        "Borrowed money (including FHLB advances, already inside othbor) plus brokered "
        "deposits over assets; funding that reprices or runs under stress.",
    ),
    _f(
        "construction_to_capital",
        "concentration",
        "lnrecons / rbc (fallback rbct1j + lnatres)",
        "ratio",
        "Construction and land loans relative to total risk-based capital; above 1 is the "
        "regulators' concentration threshold.",
    ),
    _f(
        "cre_to_capital",
        "concentration",
        "(lnrecons + lnremult + lnrenrot, else lnrenres) / rbc (fallback rbct1j + lnatres)",
        "ratio",
        "Commercial real estate exposure relative to capital; above 3 is the supervisory "
        "concentration threshold.",
    ),
    _f(
        "share_construction",
        "concentration",
        "lnrecons / lnlsgr",
        "ratio",
        "Construction and land development loans as a share of the loan book.",
    ),
    _f(
        "share_nonfarm_nonres",
        "concentration",
        "lnrenres / lnlsgr",
        "ratio",
        "Nonfarm nonresidential (commercial property) loans as a share of the loan book.",
    ),
    _f(
        "share_multifamily",
        "concentration",
        "lnremult / lnlsgr",
        "ratio",
        "Multifamily residential loans as a share of the loan book.",
    ),
    _f(
        "share_residential",
        "concentration",
        "lnreres / lnlsgr",
        "ratio",
        "1-4 family residential mortgages as a share of the loan book.",
    ),
    _f(
        "share_ci",
        "concentration",
        "lnci / lnlsgr",
        "ratio",
        "Commercial and industrial loans as a share of the loan book.",
    ),
    _f(
        "share_consumer",
        "concentration",
        "lncon / lnlsgr",
        "ratio",
        "Consumer loans as a share of the loan book.",
    ),
    _f(
        "share_agri",
        "concentration",
        "lnag / lnlsgr",
        "ratio",
        "Agricultural production loans as a share of the loan book.",
    ),
    _f(
        "log_assets",
        "structure",
        "log(asset)",
        "log_thousands_usd",
        "Natural log of total assets in thousands of dollars; bank size on a scale where "
        "a doubling is a constant step.",
    ),
    _f(
        "bank_age_years",
        "structure",
        "(repdte - estymd) / 365.25 days",
        "years",
        "Years since the charter was established; young (de novo) banks fail more often.",
    ),
    _f(
        "has_holding_company",
        "structure",
        "rssdhcr present",
        "flag",
        "True when the bank sits under a holding company that can inject capital.",
    ),
]

REGISTRY.extend(
    _f(
        f"bkclass_{code}",
        "structure",
        f"bkclass == '{code}'",
        "flag",
        f"One-hot charter class indicator for bkclass code {code}.",
    )
    for code in BKCLASS_CODES
)


def feature_names(prototype: str | None = "P1") -> list[str]:
    """Registered feature names in registry order, optionally filtered by prototype."""
    return [s.name for s in REGISTRY if prototype is None or s.prototype == prototype]


def specs_by_group(group: str) -> list[FeatureSpec]:
    """All registered features in one CAMELS group (``capital``, ``asset_quality`` ...)."""
    return [s for s in REGISTRY if s.camels_group == group]
