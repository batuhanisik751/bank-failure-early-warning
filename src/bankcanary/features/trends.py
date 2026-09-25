"""Trend and persistence features: how the key ratios are moving, not just where they are (P2).

A bank with a 3% noncurrent ratio that was 1% a year ago is in a different place from
one that has sat at 3% for a decade, and a bank that has lost money for six straight
quarters is burning capital it cannot rebuild. The trends are plain differences against
the value the same cert reported exactly one and four quarter-ends earlier
(``features.ytd.lag``, exact date match, so a filing gap gives NaN rather than a stale
comparison). The persistence counts look back over a fixed window of quarter-ends.

This module runs after the ratio modules and reads the ratios from ``deps["features"]``
(the columns built so far), never from the raw panel.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, spec
from bankcanary.features.ytd import KEY, lag

#: (ratio, source unit, monotone sign of the *level*). A trend inherits the sign of its
#: level: a rising noncurrent ratio is bad, rising equity is good. ``unrealized_loss_to_tier1``
#: is negative for a loss, so a rising value means a shrinking loss (sign -1).
TREND_RATIOS: list[tuple[str, str, int]] = [
    ("noncurrent_ratio", "ratio", 1),
    ("texas_ratio", "ratio", 1),
    ("roa_q", "ratio", -1),
    ("equity_to_assets", "ratio", -1),
    ("tier1_leverage", "percent", -1),
    ("brokered_share", "ratio", 1),
    ("uninsured_share", "ratio", 1),
    ("unrealized_loss_to_tier1", "ratio", -1),
]

#: Quarter-ends in the negative-ROA window and the minimum available for a count.
ROA_WINDOW = 8
ROA_MIN_AVAILABLE = 4
#: Quarter-ends in the rising-noncurrent window.
RISING_WINDOW = 4

GROUP = "trends"

SPECS: list[FeatureSpec] = []
for _name, _unit, _sign in TREND_RATIOS:
    for _n in (1, 4):
        SPECS.append(
            spec(
                f"d{_n}q_{_name}",
                GROUP,
                f"{_name} - {_name} {_n} quarter-end{'s' if _n > 1 else ''} earlier (same cert)",
                f"delta_{_unit}",
                f"Change in {_name} over the last {_n} quarter{'s' if _n > 1 else ''}; NaN "
                "when the earlier quarter-end was not reported.",
                monotone=_sign,
                prototype="P2",
            )
        )

SPECS.extend(
    [
        spec(
            "neg_roa_quarters_last_8",
            GROUP,
            "count(roa_q < 0) over the last 8 quarter-ends incl. current",
            "count",
            "Number of loss-making quarters among the last eight quarter-ends (current "
            "included), counted over the quarters the bank actually reported; NaN when "
            "fewer than four of the eight are available. Persistent losses erode capital.",
            monotone=1,
            prototype="P2",
        ),
        spec(
            "consecutive_loss_quarters",
            GROUP,
            "run length of roa_q < 0 ending at the current quarter (consecutive quarter-ends)",
            "count",
            "How many quarters in a row, ending now, the bank has lost money; a missing "
            "quarter or a missing ROA breaks the run and a profitable quarter resets it to 0.",
            monotone=1,
            prototype="P2",
        ),
        spec(
            "noncurrent_rising_quarters_last_4",
            GROUP,
            "count(d1q_noncurrent_ratio > 0) over the last 4 quarter-ends incl. current",
            "count",
            "Number of the last four quarter-ends (current included) at which the "
            "noncurrent ratio rose against the previous quarter; NaN when none of the "
            "four one-quarter changes is available. A steady climb precedes charge-offs.",
            monotone=1,
            prototype="P2",
        ),
    ]
)


def _keyed(panel: pd.DataFrame, features: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """``cert, repdte`` from the panel beside the requested feature columns, same index."""
    frame = panel[KEY].copy()
    for col in cols:
        frame[col] = features[col].astype("float64").to_numpy()
    return frame


def differences(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """``d1q_<ratio>`` / ``d4q_<ratio>`` for every entry of :data:`TREND_RATIOS`."""
    names = [name for name, _, _ in TREND_RATIOS]
    frame = _keyed(panel, features, names)
    out = pd.DataFrame(index=panel.index)
    for name in names:
        current = frame[name]
        for n in (1, 4):
            out[f"d{n}q_{name}"] = (current - lag(frame, name, n)).to_numpy()
    return out


def _window_counts(
    frame: pd.DataFrame, col: str, window: int, positive: bool
) -> tuple[pd.Series, pd.Series]:
    """Count of ``col < 0`` (or ``> 0``) and of non-missing values over ``window`` quarter-ends.

    Lag 0 is the current row; every other quarter-end is an exact-date lookup, so the
    window is measured in calendar quarters, not in rows.
    """
    hits = pd.Series(0.0, index=frame.index)
    available = pd.Series(0.0, index=frame.index)
    for k in range(window):
        value = frame[col] if k == 0 else lag(frame, col, k)
        value = pd.Series(value.to_numpy(), index=frame.index)
        hits = hits + ((value > 0) if positive else (value < 0)).astype("float64")
        available = available + value.notna().astype("float64")
    return hits, available


def negative_roa_quarters(panel: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Loss quarters among the last :data:`ROA_WINDOW`; NaN below :data:`ROA_MIN_AVAILABLE`."""
    frame = _keyed(panel, features, ["roa_q"])
    hits, available = _window_counts(frame, "roa_q", ROA_WINDOW, positive=False)
    return hits.where(available >= ROA_MIN_AVAILABLE)


def consecutive_loss_quarters(panel: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Run length of ``roa_q < 0`` ending at the current quarter-end, same cert.

    Rows are ordered by ``(cert, repdte)``; a run continues only when the previous row is
    the same cert exactly one quarter-end earlier *and* was a loss. A missing ROA counts
    as "not a loss", so it breaks the run and scores 0 itself.
    """
    frame = _keyed(panel, features, ["roa_q"])
    frame["repdte"] = pd.to_datetime(frame["repdte"]).dt.normalize()
    ordered = frame.sort_values(["cert", "repdte"], kind="stable")
    loss = (ordered["roa_q"] < 0).fillna(False).astype(bool)
    prev_date = (ordered["repdte"] - pd.offsets.QuarterEnd(1)).dt.normalize()
    contiguous = (ordered["cert"] == ordered["cert"].shift(1)) & (
        ordered["repdte"].shift(1) == prev_date
    )
    prev_loss = loss.shift(1, fill_value=False)
    start = loss & ~(contiguous & prev_loss)
    block = start.cumsum()
    run = ordered.groupby(block).cumcount() + 1
    result = run.where(loss, 0).astype("float64")
    return result.reindex(panel.index)


def noncurrent_rising_quarters(panel: pd.DataFrame, diffs: pd.DataFrame) -> pd.Series:
    """Positive ``d1q_noncurrent_ratio`` among the last :data:`RISING_WINDOW`; NaN if none known."""
    frame = _keyed(panel, diffs, ["d1q_noncurrent_ratio"])
    hits, available = _window_counts(frame, "d1q_noncurrent_ratio", RISING_WINDOW, positive=True)
    return hits.where(available >= 1)


def build(panel: pd.DataFrame, **deps: object) -> pd.DataFrame:
    """Trend and persistence features aligned to ``panel``'s index.

    ``deps["features"]`` must hold the ratios in :data:`TREND_RATIOS` (built by the
    earlier modules).
    """
    features = deps.get("features")
    if not isinstance(features, pd.DataFrame):
        raise ValueError("trends.build needs features=<frame of the ratios built so far>")
    missing = [name for name, _, _ in TREND_RATIOS if name not in features.columns]
    if missing:
        raise ValueError(f"trends.build: ratios not built yet: {missing}")
    out = differences(panel, features)
    out["neg_roa_quarters_last_8"] = negative_roa_quarters(panel, features).to_numpy()
    out["consecutive_loss_quarters"] = consecutive_loss_quarters(panel, features).to_numpy()
    out["noncurrent_rising_quarters_last_4"] = noncurrent_rising_quarters(panel, out).to_numpy()
    return out[[s.name for s in SPECS]]
