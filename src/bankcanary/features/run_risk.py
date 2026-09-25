"""Deposit-run risk: how much of the funding can leave in a day (the L in CAMELS, P2).

The 2023 failures were runs, not credit losses. Deposits above the $250,000 insurance
limit belong to depositors who lose money in a failure and therefore move first; large
time deposits are the same money on a short fuse. The FDIC's estimated uninsured
deposits (``depunins``, equal to Call Report RC-O item RCON5597) are populated for
almost every bank, so no estimation is needed here.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, safe_ratio, spec

#: Time deposits of $250,000 or more by remaining maturity (3 months or less, 3-12
#: months, 1-3 years, over 3 years).
LARGE_TIME_DEPOSIT_COLS = ["cd3les", "cd3t12", "cd1t3", "cdov3"]

SPECS: list[FeatureSpec] = [
    spec(
        "uninsured_share",
        "liquidity",
        "depunins / dep",
        "ratio",
        "Estimated uninsured deposits as a share of total deposits; the money that runs "
        "first because it is not protected in a failure.",
        monotone=1,
        prototype="P2",
    ),
    spec(
        "uninsured_to_liquid_assets",
        "liquidity",
        "depunins / (chbal + frepo + scaf)",
        "ratio",
        "Uninsured deposits per dollar of cash, fed funds sold and available-for-sale "
        "securities at fair value; above 1 a full run cannot be met without selling the "
        "held-to-maturity book.",
        monotone=1,
        prototype="P2",
    ),
    spec(
        "large_time_deposit_share",
        "liquidity",
        "(cd3les + cd3t12 + cd1t3 + cdov3) / dep",
        "ratio",
        "Time deposits of $250,000 or more as a share of total deposits; rate-sensitive "
        "money that is typically uninsured and leaves at maturity.",
        monotone=1,
        prototype="P2",
    ),
]


def large_time_deposits(panel: pd.DataFrame) -> pd.Series:
    """Sum of the four large-time-deposit maturity buckets; NaN when any bucket is missing."""
    total = panel[LARGE_TIME_DEPOSIT_COLS[0]].astype("float64")
    for col in LARGE_TIME_DEPOSIT_COLS[1:]:
        total = total + panel[col].astype("float64")
    return total


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Run-risk features aligned to ``panel``'s index (see ``SPECS`` for definitions)."""
    out = pd.DataFrame(index=panel.index)
    uninsured = panel["depunins"].astype("float64")
    out["uninsured_share"] = safe_ratio(uninsured, panel["dep"])
    liquid = (
        panel["chbal"].astype("float64")
        + panel["frepo"].astype("float64")
        + panel["scaf"].astype("float64")
    )
    out["uninsured_to_liquid_assets"] = safe_ratio(uninsured, liquid)
    out["large_time_deposit_share"] = safe_ratio(large_time_deposits(panel), panel["dep"])
    return out
