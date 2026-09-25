"""Hand-checked join of the point-in-time macro features into features_v2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary.features import build as build_mod
from bankcanary.features import macro, registry
from bankcanary.features.build import build_features
from bankcanary.features.spec import MONOTONE_TEXT
from tests.test_features import ROWS
from tests.test_features_p2 import MACRO_STATE, P2_ROWS

MACRO_NAMES = [
    "macro_unemp_rate",
    "macro_unemp_change_4q",
    "macro_hpi_change_4q",
    "macro_t10y3m",
    "macro_dgs10",
    "macro_fedfunds_change_4q",
]

D1, D2 = pd.Timestamp("2008-05-30"), pd.Timestamp("2008-08-29")


def make_panel() -> pd.DataFrame:
    """Five rows: GA twice, PR (territory), a null stalp and a date off the macro grid."""
    return pd.DataFrame(
        {
            "cert": [1, 1, 2, 3, 4],
            "repdte": pd.to_datetime(["2008-03-31", "2008-06-30"] + ["2008-06-30"] * 3),
            "stalp": ["GA", " ga", "PR", None, "GA"],
            "avail_date": [D1, D2, D2, D2, pd.Timestamp("2010-05-30")],
        },
        index=[10, 20, 30, 40, 50],
    )


def test_join_matches_state_and_national_rows_by_hand() -> None:
    panel = make_panel()
    out = macro.build(panel, macro_state=MACRO_STATE)
    assert list(out.columns) == MACRO_NAMES == [s.name for s in macro.SPECS]
    assert list(out.index) == list(panel.index)
    assert all(out[c].dtype == "float64" for c in out.columns)
    # GA rows carry the state series; the lower-cased code with spaces still matches.
    assert out["macro_unemp_rate"].tolist()[:2] == [5.3, 6.2]
    assert out["macro_unemp_change_4q"].tolist()[:2] == [0.7, 1.5]
    assert out["macro_hpi_change_4q"].tolist()[:2] == [-0.02, -0.05]
    # PR and the null-stalp row have no state series but keep the national columns.
    assert out.loc[[30, 40], list(macro.STATE_COLUMNS)].isna().all().all()
    assert out.loc[[30, 40], "macro_t10y3m"].tolist() == [2.1, 2.1]
    assert out.loc[[30, 40], "macro_dgs10"].tolist() == [3.8, 3.8]
    assert out.loc[[30, 40], "macro_fedfunds_change_4q"].tolist() == [-3.02, -3.02]
    # A date off the macro grid is NaN everywhere.
    assert out.loc[50].isna().all()


def test_national_rates_one_row_per_date() -> None:
    national = macro.national_rates(MACRO_STATE)
    assert national["avail_date"].tolist() == [D1, D2]
    assert national["t10y3m"].tolist() == [1.9, 2.1]
    assert national["avail_date"].dtype == "datetime64[ns]"


def test_join_rejects_bad_inputs() -> None:
    panel = make_panel()
    with pytest.raises(ValueError):
        macro.build(panel)
    with pytest.raises(ValueError):
        macro.build(panel.drop(columns=["avail_date"]), macro_state=MACRO_STATE)
    with pytest.raises(ValueError):
        macro.build(panel, macro_state=MACRO_STATE.drop(columns=["hpi_change_4q"]))
    doubled = pd.concat([MACRO_STATE, MACRO_STATE.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        macro.build(panel, macro_state=doubled)


def test_registry_places_macro_last_with_stated_signs() -> None:
    v2 = registry.feature_names(version="v2")
    assert v2[-6:] == MACRO_NAMES and len(v2) == 83
    assert registry.modules("v2")[-1] is macro and macro not in registry.modules("v1")
    assert registry.monotone_constraints(MACRO_NAMES) == [1, 1, -1, -1, 0, 1]
    for s in macro.SPECS:
        assert s.prototype == "P2" and s.camels_group == "macro"
        assert s.explanation.endswith(MONOTONE_TEXT[s.monotone])
    assert build_mod.column_order("v2")[-6:] == MACRO_NAMES
    assert "macro_unemp_rate" not in registry.feature_names("P1")


def test_build_features_v2_joins_macro_and_loads_it_when_absent(monkeypatch) -> None:
    panel = pd.DataFrame(ROWS).assign(**P2_ROWS)
    v2 = build_features(panel, version="v2", macro_state=MACRO_STATE)
    assert list(v2.columns)[-6:] == MACRO_NAMES
    assert v2["macro_unemp_rate"].tolist()[:2] == [5.3, 6.2]
    assert np.isnan(v2["macro_unemp_rate"].iloc[2]) and v2["macro_dgs10"].iloc[2] == 3.8
    # v1 never joins macro, so a v1 build does not try to read the table.
    monkeypatch.setattr(macro, "load_macro_state", lambda settings=None: pytest.fail("read"))
    assert "macro_dgs10" not in build_features(panel, version="v1").columns
    # v2 without a frame loads it through the module loader (patched: no data/ access).
    calls: list[object] = []
    monkeypatch.setattr(
        macro, "load_macro_state", lambda settings=None: calls.append(1) or MACRO_STATE
    )
    loaded = build_features(panel, version="v2")
    assert calls == [1]
    pd.testing.assert_frame_equal(loaded, v2)
    assert build_mod.with_macro_state("v2", {"macro_state": MACRO_STATE}) == {
        "macro_state": MACRO_STATE
    }
    assert calls == [1]
