"""Regional and community-bank structure features (P2).

Bank failures cluster geographically because local real-estate and employment cycles
do: Georgia and Florida in 2009-2011, Texas and Oklahoma in the 1980s, the West Coast in
2023. Fifty-plus state one-hots would be sparse and easy to overfit, so states are
mapped to the four Census Bureau regions; territories and other non-state codes land in
``region_other``. The mapping below is the Census Bureau's (``CENSUS_REGIONS``).
``is_community_bank`` is the FDIC's own community-bank designation (``cb``), which
combines size, business mix and geographic footprint rather than assets alone.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, spec

#: Census Bureau regions by postal state code; DC is counted with the South as Census does.
CENSUS_REGIONS: dict[str, tuple[str, ...]] = {
    "northeast": ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
    "midwest": ("IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "south": (
        *("DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"),
        *("AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"),
    ),
    "west": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"),
}

#: Any code that is not a state or DC (AS, FM, GU, MH, MP, PR, PW, VI ...).
OTHER = "other"

#: ``region_<code>`` column order.
REGION_CODES: tuple[str, ...] = (*CENSUS_REGIONS, OTHER)

STATE_TO_REGION: dict[str, str] = {
    state: region for region, states in CENSUS_REGIONS.items() for state in states
}

SPECS: list[FeatureSpec] = [
    spec(
        f"region_{code}",
        "structure",
        f"stalp in Census {code} region" if code != OTHER else "stalp not a state or DC",
        "flag",
        f"One-hot indicator for a head office in the Census Bureau {code} region"
        if code != OTHER
        else "One-hot indicator for a head office in a territory or other non-state code",
        monotone=0,
        prototype="P2",
    )
    for code in REGION_CODES
]
SPECS.append(
    spec(
        "is_community_bank",
        "structure",
        "cb",
        "flag",
        "True when the FDIC designates the bank a community bank (local funding and "
        "lending footprint); such banks fail through credit, not through market runs.",
        monotone=0,
        prototype="P2",
    )
)


def region_of(stalp: pd.Series) -> pd.Series:
    """Census region code for each state abbreviation; ``other`` for non-states, NA if missing."""
    codes = stalp.astype("string").str.strip().str.upper()
    return codes.map(STATE_TO_REGION).fillna(OTHER).where(codes.notna())


def one_hot_region(stalp: pd.Series) -> pd.DataFrame:
    """``region_<code>`` boolean columns; a missing ``stalp`` leaves every column False."""
    region = region_of(stalp)
    out = pd.DataFrame(index=stalp.index)
    for code in REGION_CODES:
        out[f"region_{code}"] = (region == code).fillna(False).astype(bool).to_numpy()
    return out


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Region one-hots and the community-bank flag aligned to ``panel``'s index."""
    out = one_hot_region(panel["stalp"])
    cb = panel["cb"] if "cb" in panel.columns else pd.Series(False, index=panel.index)
    out["is_community_bank"] = cb.astype("boolean").fillna(False).astype(bool).to_numpy()
    return out
