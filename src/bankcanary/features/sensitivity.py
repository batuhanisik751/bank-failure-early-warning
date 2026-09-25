"""Sensitivity to market risk: unrealised securities losses against capital (the S in CAMELS).

When rates rise, the bonds a bank bought at lower yields are worth less. Losses on
available-for-sale (AFS) securities are already in equity; losses on held-to-maturity
(HTM) securities are not booked at all, yet they become real the moment the bank must
sell to meet withdrawals. Silicon Valley Bank's HTM book carried an unbooked loss close
to its entire Tier 1 capital at the end of 2022. These features compare the fair value
of both books with their amortised cost (FDIC ``scaf``/``scaa`` and ``schf``/``scha``,
which equal the Call Report RC-B items to the dollar) and restate the leverage ratio as
if every unrealised loss were recognised.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, safe_ratio, spec

SPECS: list[FeatureSpec] = [
    spec(
        "afs_unrealized_to_tier1",
        "sensitivity",
        "(scaf - scaa) / rbct1j",
        "ratio",
        "Unrealised gain (positive) or loss (negative) on available-for-sale securities "
        "relative to Tier 1 capital; already in book equity but a marker of rate exposure.",
        monotone=-1,
        prototype="P2",
    ),
    spec(
        "htm_unrealized_to_tier1",
        "sensitivity",
        "(schf - scha) / rbct1j",
        "ratio",
        "Unrealised gain or loss on held-to-maturity securities relative to Tier 1 capital; "
        "not booked anywhere, so a large negative value is hidden capital erosion.",
        monotone=-1,
        prototype="P2",
    ),
    spec(
        "unrealized_loss_to_tier1",
        "sensitivity",
        "((scaf - scaa) + (schf - scha)) / rbct1j",
        "ratio",
        "Total unrealised securities gain or loss over Tier 1 capital; negative means a loss "
        "and a value near -1 means the losses would wipe out Tier 1 if realised.",
        monotone=-1,
        prototype="P2",
    ),
    spec(
        "adjusted_tier1_leverage",
        "capital",
        "(rbct1j + min(0, (scaf - scaa) + (schf - scha))) / asset * 100",
        "percent",
        "Tier 1 leverage recomputed after deducting unrealised AFS and HTM losses (gains are "
        "not added); the denominator is period-end total assets rather than the regulatory "
        "quarterly average, so it differs slightly from the published ratio even with no "
        "losses.",
        monotone=-1,
        prototype="P2",
    ),
    spec(
        "securities_to_assets",
        "sensitivity",
        "sc / asset",
        "ratio",
        "Securities as a share of total assets; the part of the balance sheet whose value "
        "moves with market rates.",
        monotone=0,
        prototype="P2",
    ),
    spec(
        "htm_share_of_securities",
        "sensitivity",
        "scha / sc",
        "ratio",
        "Held-to-maturity share of the securities book; HTM losses are unbooked and the "
        "bonds cannot be sold without tainting the whole portfolio.",
        monotone=1,
        prototype="P2",
    ),
]


def unrealized(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """AFS and HTM unrealised gain (+) or loss (-) in thousands of dollars.

    Either component is NaN when its fair value or amortised cost is missing (about 10%
    of rows before 2014, mostly former thrift filers), so the ratios stay missing rather
    than pretending the book has no loss.
    """
    afs = panel["scaf"].astype("float64") - panel["scaa"].astype("float64")
    htm = panel["schf"].astype("float64") - panel["scha"].astype("float64")
    return afs, htm


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Sensitivity features aligned to ``panel``'s index (see ``SPECS`` for definitions)."""
    afs, htm = unrealized(panel)
    total = afs + htm
    tier1 = panel["rbct1j"].astype("float64")
    out = pd.DataFrame(index=panel.index)
    out["afs_unrealized_to_tier1"] = safe_ratio(afs, tier1)
    out["htm_unrealized_to_tier1"] = safe_ratio(htm, tier1)
    out["unrealized_loss_to_tier1"] = safe_ratio(total, tier1)
    adjusted = tier1 + total.clip(upper=0.0)
    out["adjusted_tier1_leverage"] = safe_ratio(adjusted, panel["asset"]) * 100.0
    out["securities_to_assets"] = safe_ratio(panel["sc"], panel["asset"])
    out["htm_share_of_securities"] = safe_ratio(panel["scha"], panel["sc"])
    return out
