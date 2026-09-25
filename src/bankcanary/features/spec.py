"""The feature specification and the shared ratio guards every feature module uses.

Kept apart from :mod:`bankcanary.features.registry` so the group modules can import
these primitives while the registry imports the modules to assemble their ``SPECS``
lists (CONTRACT sections 7 and 12) without a circular import.
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

#: Plain-English name of each charter class, used in the one-hot explanations.
BKCLASS_NAMES: dict[str, str] = {
    "N": "a national bank supervised by the OCC",
    "NM": "a state-chartered bank that is not a Federal Reserve member",
    "SM": "a state-chartered bank that is a Federal Reserve member",
    "SB": "a savings bank",
    "SI": "a stock savings institution",
    "SL": "a savings and loan association",
    "NC": "a non-insured commercial bank",
    "OI": "another kind of insured institution",
}

#: Ratios that would be infinite are capped here (Texas ratio, reserve coverage).
RATIO_CAP = 10.0

#: Sentence appended to every explanation so the monotone sign is stated in words.
MONOTONE_TEXT = {
    1: "Failure risk rises with the value.",
    -1: "Failure risk falls as the value rises.",
    0: "No monotone direction is assumed.",
}


@dataclass(frozen=True)
class FeatureSpec:
    """One registered feature: what it is, how it is computed and why it matters.

    ``monotone`` is the direction a gradient-boosting constraint may enforce: ``+1`` when
    failure risk rises with the value, ``-1`` when it falls, ``0`` when the relation is
    not monotone or unknown (flags, one-hots, loan-mix shares, growth in both tails).
    """

    name: str
    camels_group: str
    formula: str
    unit: str
    explanation: str
    prototype: str = "P1"
    monotone: int = 0

    def __post_init__(self) -> None:
        if self.monotone not in MONOTONE_TEXT:
            raise ValueError(f"{self.name}: monotone must be -1, 0 or +1, got {self.monotone!r}")


def spec(
    name: str,
    group: str,
    formula: str,
    unit: str,
    explanation: str,
    monotone: int,
    prototype: str = "P1",
) -> FeatureSpec:
    """Build a :class:`FeatureSpec` with the monotone sign written into the explanation."""
    if monotone not in MONOTONE_TEXT:
        raise ValueError(f"{name}: monotone must be -1, 0 or +1, got {monotone!r}")
    text = explanation.rstrip()
    if not text.endswith("."):
        text += "."
    return FeatureSpec(
        name, group, formula, unit, f"{text} {MONOTONE_TEXT[monotone]}", prototype, monotone
    )


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
