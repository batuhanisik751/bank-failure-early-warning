"""Liquidity and funding features (the L in CAMELS).

A bank fails when it cannot meet withdrawals, so these ratios measure how much of the
funding is flighty (brokered, borrowed) and how much of the asset side can be turned
into cash quickly.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.registry import safe_ratio


def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """Liquidity features aligned to ``panel``'s index (see registry for definitions).

    ``wholesale_funding_ratio`` adds ``othbor`` and ``bro`` only: the FDIC's "other
    borrowed money" already includes FHLB advances (``othbfhlb``), so adding the
    advances again would double count them.
    """
    out = pd.DataFrame(index=panel.index)
    out["brokered_share"] = safe_ratio(panel["bro"], panel["dep"])
    out["loans_to_deposits"] = safe_ratio(panel["lnlsnet"], panel["dep"])
    liquid = panel["chbal"].astype("float64") + panel["frepo"] + panel["sc"]
    out["liquid_assets_ratio"] = safe_ratio(liquid, panel["asset"])
    wholesale = panel["othbor"].astype("float64") + panel["bro"]
    out["wholesale_funding_ratio"] = safe_ratio(wholesale, panel["asset"])
    return out
