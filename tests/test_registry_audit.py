"""Audit of the feature registry and of the generated feature documentation (step D5).

Every registered feature must be documented well enough for the UI (formula, unit, a
plain-English explanation, a monotone sign), the CONTRACT section 12 names must all be
registered as P2, ``build_features(version="v2")`` must return exactly the registry, no
feature may depend on an identifier, and ``docs/FEATURES.md`` must be the current output
of ``scripts/write_feature_docs.py``. Tests never read ``data/``.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bankcanary.features import registry
from bankcanary.features.build import build_features
from bankcanary.features.spec import MONOTONE_TEXT
from tests.test_features import ROWS
from tests.test_features_p2 import MACRO_STATE, P2_ROWS

ROOT = Path(__file__).resolve().parents[1]

#: CONTRACT section 12, verbatim.
TREND_BASES = [
    "noncurrent_ratio",
    "texas_ratio",
    "roa_q",
    "equity_to_assets",
    "tier1_leverage",
    "brokered_share",
    "uninsured_share",
    "unrealized_loss_to_tier1",
]
CONTRACT_P2_NAMES = [
    *("afs_unrealized_to_tier1", "htm_unrealized_to_tier1", "unrealized_loss_to_tier1"),
    *("adjusted_tier1_leverage", "securities_to_assets", "htm_share_of_securities"),
    *("uninsured_share", "uninsured_to_liquid_assets", "large_time_deposit_share"),
    *[f"d{n}q_{base}" for base in TREND_BASES for n in (1, 4)],
    *("neg_roa_quarters_last_8", "consecutive_loss_quarters", "noncurrent_rising_quarters_last_4"),
    *[f"region_{code}" for code in ("northeast", "midwest", "south", "west", "other")],
    "is_community_bank",
    *("macro_unemp_rate", "macro_unemp_change_4q", "macro_hpi_change_4q"),
    *("macro_t10y3m", "macro_fedfunds_change_4q"),
]

#: Columns that identify a bank rather than describe it; no formula may use them.
IDENTIFIERS = {"cert", "name", "fed_rssd", "ultcert", "newcert", "city", "zip", "webaddr"}

#: Raw FDIC field codes that are also ordinary English words and may appear in prose.
CODE_WORDS_ALLOWED = {"asset", "assets", "cb"}


def _load_script():
    path = ROOT / "scripts" / "write_feature_docs.py"
    spec = importlib.util.spec_from_file_location("write_feature_docs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def docs_script():
    return _load_script()


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(ROWS).assign(**P2_ROWS)


def _body(s) -> str:
    tail = MONOTONE_TEXT[s.monotone]
    assert s.explanation.endswith(tail), s.name
    return s.explanation[: -len(tail)].strip()


def _field_codes() -> set[str]:
    fields = yaml.safe_load((ROOT / "config" / "fields.yaml").read_text())["fields"]
    return {f["column"] for f in fields}


def test_every_spec_is_fully_documented() -> None:
    names = [s.name for s in registry.REGISTRY]
    assert len(names) == len(set(names)), "duplicate feature names"
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*|bkclass_[A-Z]+", n) for n in names)
    for s in registry.REGISTRY:
        assert s.formula.strip(), s.name
        assert s.unit.strip() and " " not in s.unit, s.name
        assert s.camels_group.strip(), s.name
        assert s.prototype in {"P1", "P2"}, s.name
        assert s.monotone in {-1, 0, 1}, s.name
        body = _body(s)
        assert len(body) >= 40 and (body[0].isupper() or body[0].isdigit()), body
        assert body.endswith("."), s.name


def test_explanations_are_plain_english_not_field_codes() -> None:
    """A non-banker reads the explanation; raw Call Report codes belong in the formula."""
    codes = _field_codes() - CODE_WORDS_ALLOWED
    feature_names = set(registry.feature_names(None))
    for s in registry.REGISTRY:
        tokens = set(re.findall(r"[a-z][a-z0-9_]*", _body(s)))
        leaked = (tokens & codes) - feature_names
        assert not leaked, f"{s.name} explanation uses raw codes {sorted(leaked)}"


def test_contract_section_12_names_are_registered_as_p2() -> None:
    by_name = {s.name: s for s in registry.REGISTRY}
    missing = [n for n in CONTRACT_P2_NAMES if n not in by_name]
    assert not missing, missing
    wrong = [n for n in CONTRACT_P2_NAMES if by_name[n].prototype != "P2"]
    assert not wrong, wrong
    assert len(CONTRACT_P2_NAMES) == 39  # + macro_dgs10 registered beside them (DECISIONS)
    assert set(registry.feature_names("P2")) == set(CONTRACT_P2_NAMES) | {"macro_dgs10"}
    assert len(registry.feature_names("P1")) == 43 and len(registry.REGISTRY) == 83


def test_build_features_v2_returns_exactly_the_registry(panel: pd.DataFrame) -> None:
    out = build_features(panel, version="v2", macro_state=MACRO_STATE)
    names = registry.feature_names(version="v2")
    assert [c for c in out.columns if c in names] == names
    assert list(out.columns) == ["cert", "repdte"] + names[:43] + ["ytd_prev_missing"] + names[43:]
    assert set(out.columns) - set(names) == {"cert", "repdte", "ytd_prev_missing"}
    assert len(out) == len(panel)
    signs = registry.monotone_constraints(names)
    assert len(signs) == len(names) and set(signs) <= {-1, 0, 1}


def test_no_formula_uses_an_identifier() -> None:
    # "(same cert)" in the trend formulas is the grouping note: the lag is taken within a
    # bank, never from the certificate number itself.
    for s in registry.REGISTRY:
        formula = s.formula.replace("(same cert)", "")
        tokens = set(re.findall(r"[a-z][a-z0-9_]*", formula))
        assert not tokens & IDENTIFIERS, f"{s.name}: {s.formula}"
    # ``has_holding_company`` only tests the presence of the parent's id, never its value.
    by_name = {s.name: s for s in registry.REGISTRY}
    assert by_name["has_holding_company"].formula == "rssdhcr present"


def test_features_do_not_depend_on_identifiers(panel: pd.DataFrame) -> None:
    """Relabelling every cert bijectively, renaming banks and shuffling rows changes nothing."""
    base = build_features(panel, version="v2", macro_state=MACRO_STATE)
    rng = np.random.default_rng(7)
    order = rng.permutation(len(panel))
    relabelled = panel.assign(cert=panel["cert"] * 7919 + 100_003, name="Some Other Bank")
    shuffled = relabelled.iloc[order].reset_index(drop=True)
    out = build_features(shuffled, version="v2", macro_state=MACRO_STATE)
    out = out.assign(cert=(out["cert"] - 100_003) // 7919)
    key = ["cert", "repdte"]
    left = base.sort_values(key).reset_index(drop=True)
    right = out.sort_values(key).reset_index(drop=True)[left.columns]
    pd.testing.assert_frame_equal(left, right)


def test_feature_docs_are_generated_and_in_sync(docs_script) -> None:
    text = docs_script.render_feature_docs()
    assert text.startswith("# Feature documentation")
    assert "## What CAMELS is" in text
    for group in docs_script.group_order(registry.REGISTRY):
        assert f"## {docs_script.GROUP_TITLES[group]}" in text
    for s in registry.REGISTRY:
        assert f"| `{s.name}` | `" in text, s.name
        assert docs_script.explanation_body(s) in text, s.name
        assert not docs_script.explanation_body(s).endswith(MONOTONE_TEXT[s.monotone])
    texas = (
        "| `texas_ratio` | `(nclnls + ore) / (eq - intan + lnatres), capped at 10` | ratio | +1 |"
    )
    assert texas in text
    committed = (ROOT / "docs" / "FEATURES.md").read_text()
    assert committed == text, "docs/FEATURES.md is stale: run scripts/write_feature_docs.py"


def test_summary_renders_from_synthetic_frames(docs_script, panel: pd.DataFrame) -> None:
    features = build_features(panel, version="v2", macro_state=MACRO_STATE)
    labels = features[["cert", "repdte"]].assign(
        y_4q=[1, 0, 0],
        label_complete_4q=[True, True, False],
        dropped_failed_before_avail=[False, False, False],
    )
    stats = docs_script.summarise_features(features, registry.feature_names(version="v2"))
    assert list(stats.columns) == ["n", "nan_share", "p1", "p50", "p99"]
    assert stats.loc["equity_to_assets", "n"] == 3 and stats.loc["region_south", "p99"] == 1.0
    medians = docs_script.p2_medians(features, labels)
    assert list(medians.columns) == ["median_y0", "median_y1"]
    assert medians.attrs["counts"] == {0: 1, 1: 1}  # the label-incomplete row is dropped
    assert medians.loc["unrealized_loss_to_tier1", "median_y1"] == pytest.approx(-0.8)
    text = docs_script.render_summary(features, labels)
    assert "1 rows with y=1 and 1 with y=0" in text and "| `macro_t10y3m` |" in text
    assert "features: 83 (43 P1 + 40 P2)" in text
