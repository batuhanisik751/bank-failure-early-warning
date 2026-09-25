"""Capital adequacy features (the C in CAMELS).

Capital is the loss-absorbing cushion between a bank's asset losses and its depositors.
The regulatory ratios (``rbc1aaj``, ``rbcrwaj``) are used as published; the book-value
ratios are recomputed from dollar amounts so the guards in the registry apply.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, safe_ratio, spec

SPECS: list[FeatureSpec] = [
    spec(
        "equity_to_assets",
        "capital",
        "eq / asset",
        "ratio",
        "Share of the balance sheet funded by shareholders; the cushion that absorbs losses "
        "before depositors are hit.",
        monotone=-1,
    ),
    spec(
        "tier1_leverage",
        "capital",
        "rbc1aaj",
        "percent",
        "Regulatory Tier 1 leverage ratio; below 4% a bank is undercapitalised and below 2% "
        "it is critically undercapitalised.",
        monotone=-1,
    ),
    spec(
        "total_rbc_ratio",
        "capital",
        "rbcrwaj",
        "percent",
        "Total risk-based capital over risk-weighted assets; missing for banks that elect the "
        "community bank leverage ratio from 2020.",
        monotone=-1,
    ),
    spec(
        "total_rbc_ratio_missing",
        "capital",
        "rbcrwaj is missing",
        "flag",
        "True when the risk-based ratio is not reported, so the model can tell a CBLR filer "
        "from a genuinely low ratio.",
        monotone=0,
    ),
    spec(
        "tangible_equity_to_assets",
        "capital",
        "(eq - intan) / (asset - intan)",
        "ratio",
        "Equity after stripping goodwill and other intangibles, which are worthless in a "
        "resolution.",
        monotone=-1,
    ),
]


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Capital features aligned to ``panel``'s index (see registry for definitions)."""
    intan = panel["intan"].astype("float64").fillna(0.0)
    out = pd.DataFrame(index=panel.index)
    out["equity_to_assets"] = safe_ratio(panel["eq"], panel["asset"])
    out["tier1_leverage"] = panel["rbc1aaj"].astype("float64")
    # The API reports an unfiled risk-based ratio as 0, not null (every CBLR filer from
    # 2020 and a few dozen rows a year before), so exactly zero means "not reported".
    # A negative ratio is a real, deeply insolvent bank and is kept as is.
    raw = panel["rbcrwaj"].astype("float64")
    rbc_ratio = raw.where(raw != 0)
    out["total_rbc_ratio"] = rbc_ratio
    out["total_rbc_ratio_missing"] = rbc_ratio.isna().to_numpy()
    out["tangible_equity_to_assets"] = safe_ratio(panel["eq"] - intan, panel["asset"] - intan)
    return out
