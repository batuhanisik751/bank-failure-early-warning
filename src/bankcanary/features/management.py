"""Management proxies: balance-sheet and loan growth (the M in CAMELS).

Management quality is not observable in the Call Report, so growth is used as its
proxy: banks that expand far faster than peers tend to be loosening underwriting, while
a shrinking bank may be in run-off. Growth is a log ratio against the value exactly
``n`` quarter-ends earlier for the same cert (``features.ytd.lag``), so filing gaps give
NaN rather than a stale comparison. The computation lives in :func:`structure.growth`;
this module only fixes the registry position (between asset quality and earnings, the
Prototype 1 column order).
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, spec
from bankcanary.features.structure import growth

SPECS: list[FeatureSpec] = [
    spec(
        "asset_growth_4q",
        "management",
        "log(asset / asset 4 quarters earlier)",
        "log_ratio",
        "One-year balance-sheet growth; rapid growth is a classic management red flag, but "
        "shrinkage can signal run-off, so both tails carry risk.",
        monotone=0,
    ),
    spec(
        "asset_growth_12q",
        "management",
        "log(asset / asset 12 quarters earlier)",
        "log_ratio",
        "Three-year balance-sheet growth, capturing a sustained expansion strategy; both "
        "tails carry risk.",
        monotone=0,
    ),
    spec(
        "loan_growth_4q",
        "management",
        "log(lnlsgr / lnlsgr 4 quarters earlier)",
        "log_ratio",
        "One-year loan-book growth; fast lending growth often means loosened underwriting, "
        "while a shrinking book can mean distress.",
        monotone=0,
    ),
]


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Growth features aligned to ``panel``'s index (see ``SPECS`` for definitions)."""
    return growth(panel)
