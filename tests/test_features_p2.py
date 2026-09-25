"""Hand-computed checks for the Prototype 2 rate and run-risk features and the registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from bankcanary.features import build as build_mod
from bankcanary.features import management, registry, run_risk, sensitivity, structure
from bankcanary.features.build import build_features, column_order
from bankcanary.features.spec import MONOTONE_TEXT, FeatureSpec, spec
from tests.test_features import ROWS

# Three rows: a bank with a large hidden loss (row 0), a bank with a small gain (row 1)
# and a bank with missing / zero denominators (row 2). Amounts in thousands.
P2_ROWS = {
    "scaa": [1000.0, 200.0, np.nan],
    "scaf": [800.0, 210.0, 50.0],
    "scha": [3000.0, 0.0, 0.0],
    "schf": [2400.0, 0.0, 0.0],
    "rbct1j": [1000.0, 50.0, 0.0],
    "asset": [10000.0, 1000.0, 500.0],
    "sc": [4000.0, 200.0, 0.0],
    "depunins": [6000.0, 300.0, 100.0],
    "dep": [8000.0, 900.0, 0.0],
    "chbal": [100.0, 40.0, 10.0],
    "frepo": [100.0, 10.0, 0.0],
    "cd3les": [400.0, 10.0, 0.0],
    "cd3t12": [200.0, 10.0, np.nan],
    "cd1t3": [100.0, 5.0, 0.0],
    "cdov3": [100.0, 2.0, 0.0],
    "stalp": ["GA", "GA", "PR"],
    "cb": [True, True, False],
    "avail_date": pd.to_datetime(["2008-05-30", "2008-08-29", "2008-08-29"]),
}

#: A tiny ``macro_state`` for the P2 rows above (GA has state series, PR only national).
MACRO_STATE = pd.DataFrame(
    {
        "stalp": ["GA", "GA", "PR", "PR"],
        "avail_date": pd.to_datetime(["2008-05-30", "2008-08-29"] * 2),
        "unemp_rate": [5.3, 6.2, np.nan, np.nan],
        "unemp_change_4q": [0.7, 1.5, np.nan, np.nan],
        "hpi_change_4q": [-0.02, -0.05, np.nan, np.nan],
        "fedfunds": [1.98, 2.0, 1.98, 2.0],
        "fedfunds_change_4q": [-3.27, -3.02, -3.27, -3.02],
        "t10y3m": [1.9, 2.1, 1.9, 2.1],
        "dgs10": [3.9, 3.8, 3.9, 3.8],
    }
)


@pytest.fixture
def panel() -> pd.DataFrame:
    frame = pd.DataFrame(ROWS)
    return frame.assign(**P2_ROWS)


def test_sensitivity_formulas(panel: pd.DataFrame) -> None:
    out = sensitivity.build(panel)
    assert out["afs_unrealized_to_tier1"].iloc[0] == pytest.approx(-0.2)
    assert out["htm_unrealized_to_tier1"].iloc[0] == pytest.approx(-0.6)
    assert out["unrealized_loss_to_tier1"].iloc[0] == pytest.approx(-0.8)
    # (1000 - 800) / 10000 * 100 = 2%; losses deducted, denominator is period-end assets.
    assert out["adjusted_tier1_leverage"].iloc[0] == pytest.approx(2.0)
    assert out["securities_to_assets"].iloc[0] == pytest.approx(0.4)
    assert out["htm_share_of_securities"].iloc[0] == pytest.approx(0.75)
    # A gain is not added back to adjusted leverage: 50 / 1000 * 100 = 5%.
    assert out["afs_unrealized_to_tier1"].iloc[1] == pytest.approx(0.2)
    assert out["unrealized_loss_to_tier1"].iloc[1] == pytest.approx(0.2)
    assert out["adjusted_tier1_leverage"].iloc[1] == pytest.approx(5.0)
    assert out["htm_share_of_securities"].iloc[1] == pytest.approx(0.0)


def test_sensitivity_guards(panel: pd.DataFrame) -> None:
    out = sensitivity.build(panel)
    row = out.iloc[2]
    # Missing amortised cost -> unrealised amounts unknown; zero Tier 1 -> NaN ratios;
    # zero securities -> no HTM share.
    for col in [
        "afs_unrealized_to_tier1",
        "htm_unrealized_to_tier1",
        "unrealized_loss_to_tier1",
        "adjusted_tier1_leverage",
        "htm_share_of_securities",
    ]:
        assert np.isnan(row[col]), col
    assert row["securities_to_assets"] == 0.0
    negative = panel.assign(rbct1j=[-5.0, 50.0, 0.0])
    assert np.isnan(sensitivity.build(negative)["unrealized_loss_to_tier1"].iloc[0])


def test_run_risk_formulas_and_guards(panel: pd.DataFrame) -> None:
    out = run_risk.build(panel)
    assert out["uninsured_share"].iloc[0] == pytest.approx(0.75)
    # 6000 / (100 + 100 + 800): AFS at fair value, HTM excluded.
    assert out["uninsured_to_liquid_assets"].iloc[0] == pytest.approx(6.0)
    assert out["large_time_deposit_share"].iloc[0] == pytest.approx(0.1)
    assert out["uninsured_share"].iloc[1] == pytest.approx(300 / 900)
    assert out["uninsured_to_liquid_assets"].iloc[1] == pytest.approx(300 / 260)
    assert out["large_time_deposit_share"].iloc[1] == pytest.approx(27 / 900)
    # Zero deposits -> NaN shares; a missing maturity bucket -> NaN large-time share.
    assert np.isnan(out["uninsured_share"].iloc[2])
    assert np.isnan(out["large_time_deposit_share"].iloc[2])
    assert out["uninsured_to_liquid_assets"].iloc[2] == pytest.approx(100 / 60)
    frame = panel.assign(cd3t12=[200.0, 10.0, 0.0], dep=[8000.0, 900.0, 50.0])
    assert run_risk.build(frame)["large_time_deposit_share"].iloc[2] == 0.0
    no_liquid = panel.assign(chbal=[0.0] * 3, frepo=[0.0] * 3, scaf=[0.0, 0.0, np.nan])
    assert run_risk.build(no_liquid)["uninsured_to_liquid_assets"].isna().all()


P2_NAMES = [
    "afs_unrealized_to_tier1",
    "htm_unrealized_to_tier1",
    "unrealized_loss_to_tier1",
    "adjusted_tier1_leverage",
    "securities_to_assets",
    "htm_share_of_securities",
    "uninsured_share",
    "uninsured_to_liquid_assets",
    "large_time_deposit_share",
]


def test_registry_versions_and_monotone_signs() -> None:
    v1 = registry.feature_names(version="v1")
    v2 = registry.feature_names(version="v2")
    assert v1 == registry.feature_names("P1") and len(v1) == 43
    assert v2[: len(v1) + len(P2_NAMES)] == v1 + P2_NAMES  # later modules append
    assert registry.feature_names(None) == v2
    assert [s.name for s in sensitivity.SPECS + run_risk.SPECS] == P2_NAMES
    assert all(s.prototype == "P2" for s in sensitivity.SPECS + run_risk.SPECS)
    for module in registry.modules("v2"):
        assert [s.name for s in module.SPECS] and callable(module.build)
    for s in registry.REGISTRY:
        assert s.monotone in (-1, 0, 1)
        assert s.explanation.endswith(MONOTONE_TEXT[s.monotone]), s.name
    signs = dict(zip(v2, registry.monotone_constraints(v2)))
    assert signs["unrealized_loss_to_tier1"] == -1 and signs["uninsured_share"] == 1
    assert signs["texas_ratio"] == 1 and signs["equity_to_assets"] == -1
    assert signs["bkclass_N"] == 0 and signs["asset_growth_4q"] == 0
    with pytest.raises(ValueError):
        registry.modules("v3")
    with pytest.raises(ValueError):
        spec("x", "capital", "f", "ratio", "Text", monotone=2)
    assert isinstance(spec("x", "capital", "f", "ratio", "Text", monotone=1), FeatureSpec)


def test_build_features_v2_extends_v1_layout(panel: pd.DataFrame) -> None:
    v1 = build_features(panel)
    v2 = build_features(panel, version="v2", macro_state=MACRO_STATE)
    assert list(v1.columns) == ["cert", "repdte"] + registry.feature_names() + ["ytd_prev_missing"]
    assert list(v2.columns)[: len(v1.columns) + len(P2_NAMES)] == list(v1.columns) + P2_NAMES
    assert list(v2.columns) == ["cert", "repdte"] + column_order("v2")
    pd.testing.assert_frame_equal(v2[v1.columns], v1)
    assert v2["unrealized_loss_to_tier1"].iloc[0] == pytest.approx(-0.8)
    # Growth columns keep their P1 position (between asset quality and earnings).
    assert list(v1.columns).index("asset_growth_4q") < list(v1.columns).index("roa_q")
    assert management.build(panel).columns.tolist() == [s.name for s in management.SPECS]
    assert "asset_growth_4q" not in structure.build(panel).columns
    with pytest.raises(ValueError):
        build_features(panel, version="v9")
    assert build_mod.table_name("v2") == "features_v2"


def test_build_features_v2_command_is_registered() -> None:
    from bankcanary.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0 and "build-features-v2" in result.output
