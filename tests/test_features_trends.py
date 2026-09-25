"""Hand-computed checks for the trend, persistence and regional structure features (P2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary.features import registry, structure_p2, trends
from bankcanary.features.build import build_features
from bankcanary.features.spec import MONOTONE_TEXT

# Bank 1 reports 2008Q1-2010Q4 except 2009Q2 (a filing gap); bank 2 reports two quarters.
Q1 = [
    "2008-03-31", "2008-06-30", "2008-09-30", "2008-12-31", "2009-03-31",
    "2009-09-30", "2009-12-31", "2010-03-31", "2010-06-30", "2010-09-30", "2010-12-31",
]  # fmt: skip
Q2 = ["2010-06-30", "2010-09-30"]
NC = [1, 2, 3, 2, 3, 5, 4, 6, 7, 8, 9, 1, 0.5]
ROA = [-1, -1, 1, -1, -1, -1, -1, -1, np.nan, -1, -1, -1, -1]
OTHER_RATIOS = [n for n, _, _ in trends.TREND_RATIOS if n not in ("noncurrent_ratio", "roa_q")]


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel keys + the ratio frame trends reads, rows shuffled to test index alignment."""
    panel = pd.DataFrame({"cert": [1] * len(Q1) + [2] * len(Q2), "repdte": pd.to_datetime(Q1 + Q2)})
    features = pd.DataFrame({"noncurrent_ratio": NC, "roa_q": ROA}, dtype="float64")
    for name in OTHER_RATIOS:
        features[name] = 1.0
    order = [5, 12, 0, 8, 3, 11, 10, 1, 7, 2, 9, 4, 6]
    return panel.iloc[order].reset_index(drop=True), features.iloc[order].reset_index(drop=True)


def by_row(out: pd.DataFrame, panel: pd.DataFrame, col: str) -> list[float]:
    """Values of ``col`` in the original (cert, repdte) order of ``Q1 + Q2``."""
    keyed = pd.concat([panel, out], axis=1).sort_values(["cert", "repdte"])
    return keyed[col].tolist()


def assert_seq(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert (np.isnan(a) and np.isnan(e)) or a == pytest.approx(e), (actual, expected)


def test_differences_use_exact_quarter_match() -> None:
    panel, features = make_frames()
    out = trends.build(panel, features=features)
    assert list(out.columns) == [s.name for s in trends.SPECS]
    nan = np.nan
    assert_seq(
        by_row(out, panel, "d1q_noncurrent_ratio"),
        [nan, 1, 1, -1, 1, nan, -1, 2, 1, 1, 1, nan, -0.5],
    )
    assert_seq(
        by_row(out, panel, "d4q_noncurrent_ratio"),
        [nan, nan, nan, nan, 2, 2, 2, 3, nan, 3, 5, nan, nan],
    )
    # A constant ratio changes by exactly 0 wherever the earlier quarter-end exists.
    assert_seq(
        by_row(out, panel, "d1q_uninsured_share"),
        [nan, 0, 0, 0, 0, nan, 0, 0, 0, 0, 0, nan, 0],
    )
    assert_seq(
        by_row(out, panel, "d4q_tier1_leverage"),
        [nan, nan, nan, nan, 0, 0, 0, 0, nan, 0, 0, nan, nan],
    )


def test_persistence_counts_respect_gaps_and_missing_roa() -> None:
    panel, features = make_frames()
    out = trends.build(panel, features=features)
    nan = np.nan
    # Run length resets on a profit (2008Q3), breaks at the 2009Q2 gap and on a NaN ROA.
    assert_seq(
        by_row(out, panel, "consecutive_loss_quarters"),
        [1, 2, 0, 1, 2, 1, 2, 3, 0, 1, 2, 1, 2],
    )
    # Counted over reported quarters in the 8-quarter calendar window; NaN below 4.
    assert_seq(
        by_row(out, panel, "neg_roa_quarters_last_8"),
        [nan, nan, nan, 3, 4, 5, 6, 6, 5, 6, 6, nan, nan],
    )
    # Positive one-quarter changes among the last four quarter-ends; NaN when none known.
    assert_seq(
        by_row(out, panel, "noncurrent_rising_quarters_last_4"),
        [nan, 1, 2, 2, 3, 1, 1, 1, 2, 3, 4, nan, 0],
    )


def test_trends_build_requires_ratios() -> None:
    panel, features = make_frames()
    with pytest.raises(ValueError):
        trends.build(panel)
    with pytest.raises(ValueError):
        trends.build(panel, features=features.drop(columns=["texas_ratio"]))


def test_region_mapping_covers_every_state_once() -> None:
    states = [s for group in structure_p2.CENSUS_REGIONS.values() for s in group]
    assert len(states) == 51 and len(set(states)) == 51  # 50 states + DC
    assert structure_p2.REGION_CODES == ("northeast", "midwest", "south", "west", "other")
    region = structure_p2.region_of(pd.Series(["ny", " TX ", "IL", "CA", "PR", "GU", None, "DC"]))
    assert region.tolist()[:6] == ["northeast", "south", "midwest", "west", "other", "other"]
    assert pd.isna(region.iloc[6]) and region.iloc[7] == "south"


def test_structure_p2_build_one_hots_and_community_flag() -> None:
    panel = pd.DataFrame(
        {
            "cert": [1, 2, 3, 4],
            "repdte": pd.to_datetime(["2020-03-31"] * 4),
            "stalp": ["MA", "VI", None, "wa"],
            "cb": pd.array([True, False, None, True], dtype="boolean"),
        }
    )
    out = structure_p2.build(panel)
    assert list(out.columns) == [s.name for s in structure_p2.SPECS]
    assert out["region_northeast"].tolist() == [True, False, False, False]
    assert out["region_other"].tolist() == [False, True, False, False]
    assert out["region_west"].tolist() == [False, False, False, True]
    assert not out.loc[2, [f"region_{c}" for c in structure_p2.REGION_CODES]].any()
    assert out["is_community_bank"].tolist() == [True, False, False, True]
    assert all(out[c].dtype == bool for c in out.columns)


def test_registry_order_signs_and_v2_build(panel_p2: pd.DataFrame) -> None:
    v2 = registry.feature_names(version="v2")
    new = [s.name for s in trends.SPECS + structure_p2.SPECS]
    assert v2[-len(new) :] == new and len(v2) == 77
    assert v2.index("d1q_noncurrent_ratio") > v2.index("uninsured_share")
    signs = dict(zip(v2, registry.monotone_constraints(v2)))
    assert signs["d1q_noncurrent_ratio"] == 1 and signs["d4q_texas_ratio"] == 1
    assert signs["d4q_equity_to_assets"] == -1 and signs["d1q_roa_q"] == -1
    assert signs["d4q_unrealized_loss_to_tier1"] == -1 and signs["d1q_brokered_share"] == 1
    assert signs["consecutive_loss_quarters"] == 1 and signs["neg_roa_quarters_last_8"] == 1
    assert signs["noncurrent_rising_quarters_last_4"] == 1
    assert all(signs[f"region_{c}"] == 0 for c in structure_p2.REGION_CODES)
    for s in trends.SPECS + structure_p2.SPECS:
        assert s.prototype == "P2" and s.explanation.endswith(MONOTONE_TEXT[s.monotone])
    frame = build_features(panel_p2, version="v2")
    assert list(frame.columns)[-len(new) :] == new
    assert frame["consecutive_loss_quarters"].notna().all()
    # Bank 1's Q2 row sees its Q1 row one quarter earlier; bank 2 has no history.
    assert frame["d1q_equity_to_assets"].iloc[1] == pytest.approx(0.12 - 0.01)
    assert np.isnan(frame["d1q_equity_to_assets"].iloc[2])


@pytest.fixture
def panel_p2() -> pd.DataFrame:
    from tests.test_features import ROWS
    from tests.test_features_p2 import P2_ROWS

    return pd.DataFrame(ROWS).assign(**P2_ROWS)
