"""Feature registry: the fixed order of feature modules and their specs (CONTRACT 7 and 12).

Every feature module exposes ``SPECS: list[FeatureSpec]`` (one entry per column it
produces, with the CAMELS group, formula, unit, banking explanation, prototype and
monotone direction) and ``build(panel, **deps) -> DataFrame`` returning exactly those
columns. This module fixes the order in which the modules run for each feature-table
version, so the column order of ``features_v1`` / ``features_v2`` is the order below and
the model code takes its feature list from :func:`feature_names` rather than from the
table. The primitives (``FeatureSpec``, ``safe_ratio`` ...) live in
:mod:`bankcanary.features.spec` and are re-exported here.
"""

from __future__ import annotations

from types import ModuleType

from bankcanary.features import (
    asset_quality,
    capital,
    concentration,
    earnings,
    liquidity,
    management,
    run_risk,
    sensitivity,
    structure,
)
from bankcanary.features.spec import (  # noqa: F401  (re-exported for callers)
    BKCLASS_CODES,
    RATIO_CAP,
    FeatureSpec,
    log_ratio,
    safe_ratio,
)

#: Prototype 1 modules, in build order (CONTRACT section 7).
MODULES_V1: tuple[ModuleType, ...] = (
    capital,
    asset_quality,
    management,
    earnings,
    liquidity,
    concentration,
    structure,
)

#: Prototype 2 modules appended after the P1 set (CONTRACT section 12). Later steps
#: add trends, region/community-bank structure and macro here, in that order.
MODULES_P2: tuple[ModuleType, ...] = (sensitivity, run_risk)

#: Module order per feature-table version.
MODULES_BY_VERSION: dict[str, tuple[ModuleType, ...]] = {
    "v1": MODULES_V1,
    "v2": MODULES_V1 + MODULES_P2,
}

#: Every registered feature, in build order of the widest version.
REGISTRY: list[FeatureSpec] = [s for module in MODULES_BY_VERSION["v2"] for s in module.SPECS]


def modules(version: str = "v1") -> tuple[ModuleType, ...]:
    """The feature modules that make up ``features_<version>``, in build order."""
    try:
        return MODULES_BY_VERSION[version]
    except KeyError:
        raise ValueError(f"unknown feature version {version!r}") from None


def specs(version: str = "v1") -> list[FeatureSpec]:
    """Every spec of ``features_<version>`` in column order."""
    return [s for module in modules(version) for s in module.SPECS]


def feature_names(prototype: str | None = "P1", version: str | None = None) -> list[str]:
    """Registered feature names in registry order.

    ``prototype`` filters the whole registry (``"P1"`` by default; ``None`` = all);
    ``version`` instead returns the columns of one feature table (``"v1"`` / ``"v2"``).
    """
    if version is not None:
        return [s.name for s in specs(version)]
    return [s.name for s in REGISTRY if prototype is None or s.prototype == prototype]


def specs_by_group(group: str) -> list[FeatureSpec]:
    """All registered features in one CAMELS group (``capital``, ``asset_quality`` ...)."""
    return [s for s in REGISTRY if s.camels_group == group]


def monotone_constraints(names: list[str]) -> list[int]:
    """The ``monotone`` sign of each named feature, in the given order (0 when unregistered).

    This is the vector the gradient-boosting ``gbdt_mono`` config hands to the estimator.
    """
    signs = {s.name: s.monotone for s in REGISTRY}
    return [signs.get(n, 0) for n in names]


def check_registry() -> None:
    """Raise if any name is registered twice or the modules' specs disagree with the tables."""
    names = [s.name for s in REGISTRY]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise RuntimeError(f"features registered more than once: {duplicates}")
