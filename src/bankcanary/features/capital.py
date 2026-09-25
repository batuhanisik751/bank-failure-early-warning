"""Capital adequacy features (the C in CAMELS).

Capital is the loss-absorbing cushion between a bank's asset losses and its depositors.
The regulatory ratios (``rbc1aaj``, ``rbcrwaj``) are used as published; the book-value
ratios are recomputed from dollar amounts so the guards in the registry apply.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.registry import safe_ratio


def compute(panel: pd.DataFrame) -> pd.DataFrame:
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
