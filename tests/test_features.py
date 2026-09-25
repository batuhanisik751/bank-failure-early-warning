"""Hand-computed checks for the Prototype 1 feature set, its guards and the registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary.features import (
    asset_quality,
    capital,
    concentration,
    earnings,
    liquidity,
    structure,
)
from bankcanary.features.build import build_features
from bankcanary.features.registry import BKCLASS_CODES, RATIO_CAP, REGISTRY, feature_names

# One bank (cert 1) over Q1-Q2 2008 and a second bank (cert 2) with Q2 only; every
# amount is chosen so the expected ratios are exact decimals.
ROWS = {
    "cert": [1, 1, 2],
    "repdte": pd.to_datetime(["2008-03-31", "2008-06-30", "2008-06-30"]),
    "asset": [1000.0, 1200.0, 500.0],
    "eq": [100.0, 120.0, 50.0],
    "intan": [10.0, 20.0, np.nan],
    "rbc1aaj": [9.5, 9.0, np.nan],
    "rbcrwaj": [13.0, np.nan, 12.0],
    "lnlsgr": [600.0, 800.0, 400.0],
    "lnlsnet": [590.0, 785.0, 395.0],
    "lnatres": [10.0, 15.0, 5.0],
    "nclnls": [12.0, 40.0, 0.0],
    "ore": [3.0, 10.0, 0.0],
    "p3asset": [6.0, 16.0, 4.0],
    "ntlnls": [2.0, 9.0, 4.0],
    "netinc": [5.0, 16.0, 2.0],
    "intinc": [20.0, 45.0, 10.0],
    "eintexp": [8.0, 17.0, 4.0],
    "nim": [12.0, 28.0, 6.0],
    "nonii": [2.0, 5.0, 1.0],
    "nonix": [7.0, 19.0, 21.0],
    "elnatr": [1.0, 4.0, 0.0],
    "ernast": [900.0, 1100.0, np.nan],
    "bro": [50.0, 90.0, 0.0],
    "dep": [800.0, 900.0, 0.0],
    "chbal": [40.0, 50.0, 10.0],
    "frepo": [10.0, 10.0, 0.0],
    "sc": [150.0, 180.0, 40.0],
    "othbor": [60.0, 90.0, 20.0],
    "othbfhlb": [60.0, 40.0, 20.0],
    "rbc": [110.0, 0.0, 55.0],
    "rbct1j": [100.0, 105.0, 50.0],
    "lnrecons": [110.0, 120.0, 40.0],
    "lnremult": [30.0, 40.0, 10.0],
    "lnrenres": [120.0, 160.0, 80.0],
    "lnrenrot": [np.nan, 80.0, 30.0],
    "lnreres": [180.0, 240.0, 120.0],
    "lnci": [90.0, 120.0, 60.0],
    "lncon": [30.0, 40.0, 20.0],
    "lnag": [6.0, 8.0, 4.0],
    "estymd": pd.to_datetime(["1998-03-31", "1998-03-31", "2007-12-31"]),
    "has_holding_company": [True, True, False],
    "bkclass": ["NM", "NM", "SB"],
}


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(ROWS)


def test_capital_formulas_and_missing_indicator(panel: pd.DataFrame) -> None:
    out = capital.compute(panel)
    assert out["equity_to_assets"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert out["tier1_leverage"].tolist()[:2] == [9.5, 9.0]
    assert out["total_rbc_ratio_missing"].tolist() == [False, True, False]
    # (100 - 10) / (1000 - 10); NaN intangibles count as zero.
    assert out["tangible_equity_to_assets"].iloc[0] == pytest.approx(90 / 990)
    assert out["tangible_equity_to_assets"].iloc[2] == pytest.approx(0.1)


def test_asset_quality_formulas(panel: pd.DataFrame) -> None:
    out = asset_quality.compute(panel)
    assert out["noncurrent_ratio"].tolist() == pytest.approx([0.02, 0.05, 0.0])
    assert out["npa_to_assets"].tolist() == pytest.approx([0.015, 50 / 1200, 0.0])
    assert out["early_delinquency"].tolist() == pytest.approx([0.01, 0.02, 0.01])
    # Q2 charge-offs 9 - 2 = 7, annualised 28, over average loans (600 + 800) / 2.
    assert out["nco_rate"].iloc[1] == pytest.approx(28 / 700)
    # Q1 is already one quarter: 2 * 4 / 600 (no previous quarter to average with).
    assert out["nco_rate"].iloc[0] == pytest.approx(8 / 600)
    # Cert 2 has no Q1 row: YTD / 2 quarters = 2, annualised 8, over 400.
    assert out["nco_rate"].iloc[2] == pytest.approx(8 / 400)
    assert out["reserve_coverage"].tolist()[:2] == pytest.approx([10 / 12, 15 / 40])
    # (12 + 3) / (100 - 10 + 10)
    assert out["texas_ratio"].iloc[0] == pytest.approx(0.15)
    assert not out["texas_ratio_capped"].iloc[0]


def test_asset_quality_guards() -> None:
    frame = pd.DataFrame(
        {
            "cert": [1, 2, 3],
            "repdte": pd.to_datetime(["2008-03-31"] * 3),
            "nclnls": [0.0, 30.0, 5.0],
            "ore": [0.0, 5.0, 0.0],
            "lnatres": [4.0, 2.0, 100.0],
            "eq": [50.0, 8.0, 50.0],
            "intan": [0.0, 10.0, 0.0],
            "lnlsgr": [0.0, 100.0, 100.0],
            "asset": [100.0, 100.0, 100.0],
            "p3asset": [1.0, 1.0, 1.0],
            "ntlnls": [1.0, 1.0, 1.0],
        }
    )
    out = asset_quality.compute(frame)
    # Zero loans: ratios over lnlsgr are NaN, never inf.
    assert np.isnan(out["noncurrent_ratio"].iloc[0]) and np.isnan(out["nco_rate"].iloc[0])
    # No noncurrent loans: coverage is the cap, not inf.
    assert out["reserve_coverage"].iloc[0] == RATIO_CAP
    # Reserves 20x noncurrent loans are clipped at the cap.
    assert out["reserve_coverage"].iloc[2] == RATIO_CAP
    # Tangible equity + reserves = 8 - 10 + 2 = 0 -> capped and flagged.
    assert out["texas_ratio"].iloc[1] == RATIO_CAP
    assert out["texas_ratio_capped"].tolist() == [False, True, False]


def test_earnings_formulas(panel: pd.DataFrame) -> None:
    out = earnings.compute(panel)
    # Q2 net income 16 - 5 = 11, annualised 44, over average assets 1100.
    assert out["roa_q"].iloc[1] == pytest.approx(44 / 1100)
    # Q2 net interest (45 - 20) - (17 - 8) = 16, annualised 64, over avg earning assets 1000.
    assert out["nim_q"].iloc[1] == pytest.approx(64 / 1000)
    # Cert 2 lacks ernast: falls back to (average) total assets 500; (10 - 4) / 2 * 4 = 12.
    assert out["nim_q"].iloc[2] == pytest.approx(12 / 500)
    # Q2 noninterest expense 12 over net interest income 16 + noninterest income 3.
    assert out["efficiency_ratio"].iloc[1] == pytest.approx(12 / 19)
    # Q2 provision 3, annualised 12, over average loans 700.
    assert out["provision_rate"].iloc[1] == pytest.approx(12 / 700)


def test_liquidity_formulas_and_zero_deposits(panel: pd.DataFrame) -> None:
    out = liquidity.compute(panel)
    assert out["brokered_share"].iloc[0] == pytest.approx(50 / 800)
    assert out["loans_to_deposits"].iloc[0] == pytest.approx(590 / 800)
    assert out["liquid_assets_ratio"].iloc[0] == pytest.approx(200 / 1000)
    # othbor already includes FHLB advances: (90 + 90) / 1200, not (90 + 40 + 90) / 1200.
    assert out["wholesale_funding_ratio"].iloc[1] == pytest.approx(180 / 1200)
    assert np.isnan(out["brokered_share"].iloc[2]) and np.isnan(out["loans_to_deposits"].iloc[2])


def test_concentration_capital_fallback_and_shares(panel: pd.DataFrame) -> None:
    out = concentration.compute(panel)
    # Q1: rbc = 110 and no owner-occupied split, so total nonfarm nonres is used.
    assert out["construction_to_capital"].iloc[0] == pytest.approx(110 / 110)
    assert out["cre_to_capital"].iloc[0] == pytest.approx((110 + 30 + 120) / 110)
    # Q2: rbc = 0 (CBLR filer) -> tier 1 + allowance = 120; non-owner-occupied 80.
    assert out["construction_to_capital"].iloc[1] == pytest.approx(120 / 120)
    assert out["cre_to_capital"].iloc[1] == pytest.approx((120 + 40 + 80) / 120)
    assert out["share_construction"].iloc[0] == pytest.approx(110 / 600)
    assert out["share_agri"].iloc[2] == pytest.approx(4 / 400)


def test_zero_risk_based_ratio_is_not_reported(panel: pd.DataFrame) -> None:
    frame = panel.assign(rbcrwaj=[0.0, -3.5, 12.0])
    out = capital.compute(frame)
    assert np.isnan(out["total_rbc_ratio"].iloc[0])
    assert out["total_rbc_ratio"].iloc[1] == -3.5  # negative capital is real signal
    assert out["total_rbc_ratio_missing"].tolist() == [True, False, False]


def test_structure_growth_age_and_one_hot() -> None:
    dates = pd.to_datetime(["2007-03-31", "2008-03-31", "2010-03-31"])
    frame = pd.DataFrame(
        {
            "cert": [1, 1, 1],
            "repdte": dates,
            "asset": [1000.0, 1100.0, 1210.0],
            "lnlsgr": [500.0, 600.0, 0.0],
            "estymd": pd.to_datetime(["2000-03-31"] * 3),
            "has_holding_company": [True, False, True],
            "bkclass": ["NM", "sb ", "XX"],
        }
    )
    out = structure.compute(frame)
    assert np.isnan(out["asset_growth_4q"].iloc[0])
    assert out["asset_growth_4q"].iloc[1] == pytest.approx(np.log(1.1))
    # 2010-03-31 has no row 4 quarters earlier (NaN) but is exactly 12 after 2007-03-31.
    assert np.isnan(out["asset_growth_4q"].iloc[2])
    assert out["asset_growth_12q"].iloc[2] == pytest.approx(np.log(1.21))
    assert np.isnan(out["loan_growth_4q"].iloc[2])  # lnlsgr = 0 is not a valid base
    assert out["loan_growth_4q"].iloc[1] == pytest.approx(np.log(1.2))
    assert out["log_assets"].iloc[0] == pytest.approx(np.log(1000.0))
    assert out["bank_age_years"].iloc[1] == pytest.approx(8.0, abs=0.01)
    assert out["has_holding_company"].tolist() == [True, False, True]
    onehot = out[[f"bkclass_{c}" for c in BKCLASS_CODES]]
    assert onehot.iloc[0].tolist() == [c == "NM" for c in BKCLASS_CODES]
    assert onehot.iloc[1].tolist() == [c == "SB" for c in BKCLASS_CODES]  # trimmed, upper
    assert not onehot.iloc[2].any()  # unknown code: all False
    assert onehot.dtypes.eq(bool).all()


def test_registry_is_complete_and_unique() -> None:
    names = feature_names()
    assert len(names) == len(set(names))
    assert {s.prototype for s in REGISTRY} == {"P1"}
    assert all(s.explanation.endswith(".") and s.camels_group for s in REGISTRY)
    assert names[:3] == ["equity_to_assets", "tier1_leverage", "total_rbc_ratio"]
    assert names[-len(BKCLASS_CODES) :] == [f"bkclass_{c}" for c in BKCLASS_CODES]


def test_build_features_returns_exactly_the_registry(panel: pd.DataFrame) -> None:
    out = build_features(panel)
    assert list(out.columns) == ["cert", "repdte"] + feature_names() + ["ytd_prev_missing"]
    assert len(out) == len(panel)
    assert out["ytd_prev_missing"].tolist() == [False, False, True]
    assert out["texas_ratio_capped"].dtype == bool and out["bkclass_NM"].dtype == bool
    assert out["repdte"].dtype == "datetime64[ns]"
    # Values flow through from the group modules unchanged.
    assert out["equity_to_assets"].iloc[0] == pytest.approx(0.1)
    assert out["roa_q"].iloc[1] == pytest.approx(44 / 1100)
