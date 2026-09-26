"""SHAP drivers per bank-quarter for the walk-forward gradient boosters (spec 7, 9; D10).

A score alone does not tell a supervisor *why* a bank sits near the top of the list.
For every walk-forward test year ``Y`` this module runs ``shap.TreeExplainer`` over the
year-``Y`` booster (``models/walkforward/<Y>/<model>/``, ``gbdt_mono`` since Decision
Point 2 made the monotone booster the production model) on that year's test rows, and
the most recent booster, the "production" model, over every report quarter that lies past
the last complete test year (those quarters are scored but never evaluated, because
their outcome windows are still open). The SHAP values are in the booster's log-odds
units: they add up, with the explainer's expected value, to the raw score, so the ten
rows kept per bank-quarter (five largest positive, five largest negative contributions)
say which ratios pushed that bank's odds of failure up and which pulled them down.

The winsoriser is applied before the explainer so the feature values stored next to each
contribution are the ones the trees saw. Per-year files land in ``data/drivers/`` and
:func:`rebuild_drivers_table` folds them into the ``drivers`` table (Parquet + DuckDB).
Every year logs an ``explain`` run whose metrics carry the mean absolute SHAP of each
feature; :func:`write_summary` reads those back to write ``reports/shap_summary.md``,
pooling them as the average over years of each year's *share* of the total (so a year
whose booster runs on a larger log-odds scale, such as the four-failure 2020 fit, weighs
the same as every other year).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bankcanary.config import Settings

log = logging.getLogger(__name__)

TABLE = "drivers"
TABLE_KEY: tuple[str, ...] = ("cert", "repdte", "model", "rank")
COLUMNS: tuple[str, ...] = TABLE_KEY + (
    "feature",
    "shap_value",
    "feature_value",
    "direction",
    "model_year",
)
MODEL = "gbdt_mono"
#: ``model`` value of the rows scored by the most recent booster past the backtest years.
PRODUCTION_MODEL = "gbdt_mono_production"
PRODUCTION = "production"
HORIZON = 4
TOP_N = 5
RUN_NAME = "explain"
#: Spec rule 6.6: a feature carrying more than this share of the total mean |SHAP| is flagged.
DOMINANCE_SHARE = 0.40


def shap_matrix(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float, np.ndarray]:
    """``(shap_values, expected_value, X_winsorised)`` of a ``make_gbdt`` pipeline on ``X``.

    The pipeline's fitted ``Winsorizer`` transforms ``X`` first (the explainer must see
    the clipped values the trees split on); ``TreeExplainer`` then returns one log-odds
    contribution per (row, feature). shap returns either a ``(n, p)`` array or a two-class
    list for a binary booster depending on its version; the positive class is kept.
    """
    import warnings

    import shap

    clipped = pipeline.named_steps["winsorize"].transform(X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")
        values = explainer.shap_values(clipped)
    expected = explainer.expected_value
    if isinstance(values, list):
        values, expected = values[-1], np.ravel(expected)[-1]
    elif values.ndim == 3:
        values, expected = values[:, :, -1], np.ravel(expected)[-1]
    return np.asarray(values, dtype=float), float(np.ravel(expected)[-1]), clipped


def drivers_table(
    keys: pd.DataFrame,
    values: np.ndarray,
    clipped: np.ndarray,
    features: list[str],
    model: str,
    model_year: int,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """The CONTRACT ``drivers`` rows for one block of bank-quarters.

    ``keys`` holds ``cert, repdte`` for the rows of ``values`` (``(n, p)`` SHAP values)
    and ``clipped`` (the winsorised feature values). Ranks ``1..top_n`` are the largest
    positive contributions (``direction = 'raises'``, biggest first); ranks
    ``top_n + 1 .. 2 * top_n`` the most negative ones (``'lowers'``, most negative
    first). A contribution of exactly zero is never a driver, so a bank-quarter can have
    fewer than ``2 * top_n`` rows. Ties are broken by feature order, so the table is a
    deterministic function of its inputs.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (len(keys), len(features)):
        raise ValueError(f"SHAP values {values.shape} do not match {len(keys)}x{len(features)}")
    n = len(keys)
    raises = _ranked(values, top_n, sign=1.0)
    lowers = _ranked(values, top_n, sign=-1.0)
    parts = []
    for block, direction, offset in ((raises, "raises", 0), (lowers, "lowers", top_n)):
        rows, ranks, cols = block
        if len(rows) == 0:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "cert": keys["cert"].to_numpy()[rows],
                    "repdte": keys["repdte"].to_numpy()[rows],
                    "model": model,
                    "rank": ranks + offset,
                    "feature": np.asarray(features, dtype=object)[cols],
                    "shap_value": values[rows, cols],
                    "feature_value": np.asarray(clipped, dtype=float)[rows, cols],
                    "direction": direction,
                    "model_year": int(model_year),
                }
            )
        )
    if not parts:
        return _empty_drivers()
    out = pd.concat(parts, ignore_index=True)
    out = out.astype({"rank": "int64", "model": "str", "feature": "str", "direction": "str"})
    out = out.astype({"repdte": "datetime64[ns]", "model_year": "int64"})
    log.debug("drivers for %d bank-quarters: %d rows", n, len(out))
    return out.sort_values(list(TABLE_KEY), kind="mergesort").reset_index(drop=True)


def _ranked(
    values: np.ndarray, top_n: int, sign: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row, rank (1-based) and column arrays of the ``top_n`` largest ``sign * values`` > 0."""
    signed = sign * values
    order = np.argsort(-signed, axis=1, kind="stable")[:, :top_n]
    picked = np.take_along_axis(signed, order, axis=1) > 0
    rows, positions = np.nonzero(picked)
    return rows, positions + 1, order[rows, positions]


def _empty_drivers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cert": pd.Series(dtype="int64"),
            "repdte": pd.Series(dtype="datetime64[ns]"),
            "model": pd.Series(dtype="str"),
            "rank": pd.Series(dtype="int64"),
            "feature": pd.Series(dtype="str"),
            "shap_value": pd.Series(dtype="float64"),
            "feature_value": pd.Series(dtype="float64"),
            "direction": pd.Series(dtype="str"),
            "model_year": pd.Series(dtype="int64"),
        }
    )


def drivers_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "drivers"


def production_model(model: str = MODEL) -> str:
    """``model`` label of the rows the latest ``model`` booster scores past the backtest."""
    return f"{model}_production"


def drivers_path(settings: Settings, year: int | str, model: str = MODEL) -> Path:
    """``data/drivers/<Y>_<model>.parquet`` (``production_<model>.parquet`` for the latest rows)."""
    return drivers_dir(settings) / f"{year}_{model}.parquet"


def production_rows(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Rows past the last complete test year: scored by the production model, never evaluated.

    Every report quarter after the latest year with a complete ``horizon``-quarter label,
    minus the post-failure reports (``dropped_failed_before_avail``), whether or not its
    own label is complete: these are the quarters a supervisor would be looking at today.
    """
    from bankcanary.evaluation.walkforward import latest_complete_year
    from bankcanary.splits.time_split import DROPPED_COL

    last = latest_complete_year(frame, horizon)
    dropped = frame[DROPPED_COL].fillna(True).astype(bool)
    mask = (frame["repdte"].dt.year > last) & ~dropped
    return mask.rename(f"production_{horizon}q")


def select_rows(
    frame: pd.DataFrame, year: int | str, horizon: int = HORIZON
) -> tuple[pd.Series, int]:
    """``(mask, model_year)``: which rows to explain and which booster explains them.

    A test year uses its own walk-forward booster on its own test rows
    (:func:`bankcanary.evaluation.walkforward.year_masks`); ``'production'`` uses the
    most recent walk-forward booster on :func:`production_rows`.
    """
    from bankcanary.evaluation.walkforward import latest_complete_year, year_masks

    if year == PRODUCTION:
        return production_rows(frame, horizon), latest_complete_year(frame, horizon)
    _, test = year_masks(frame, horizon, int(year))
    return test, int(year)


def explain_year(
    frame: pd.DataFrame, settings: Settings, year: int | str, save: bool = True, model: str = MODEL
):
    """SHAP drivers of one test year (or ``'production'``) and the run record that logs it.

    Returns ``(drivers, mean_abs)`` where ``mean_abs`` is the mean |SHAP| per feature over
    the explained rows (a ``Series`` indexed by feature, descending). ``model`` names the
    walk-forward booster family (``gbdt_mono`` or ``gbdt``). The run config is
    ``{model, horizon, year, model_year, features_version}`` so a re-run of the same year
    and model overwrites the same ``runs/explain/`` record.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.walkforward import FEATURE_VERSION, load_year

    mask, model_year = select_rows(frame, year, HORIZON)
    if not mask.any():
        raise ValueError(f"no rows to explain for {year}")
    pipeline, features, _ = load_year(settings, model_year, model, HORIZON)
    rows = frame.loc[mask]
    values, expected, clipped = shap_matrix(pipeline, rows[list(features)])
    label = production_model(model) if year == PRODUCTION else model
    drivers = drivers_table(rows[["cert", "repdte"]], values, clipped, features, label, model_year)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=features, name="mean_abs_shap")
    mean_abs = mean_abs.sort_values(ascending=False, kind="mergesort")
    share = float(mean_abs.iloc[0] / mean_abs.sum()) if mean_abs.sum() > 0 else 0.0
    config = {
        "model": model,
        "horizon": HORIZON,
        "year": str(year),
        "model_year": int(model_year),
        "features_version": FEATURE_VERSION,
    }
    if save:
        path = drivers_path(settings, year, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        drivers.loc[:, list(COLUMNS)].to_parquet(path, index=False)
        run = tracking.start_run(RUN_NAME, config, settings)
        run.log_metrics(
            {
                "n_rows": int(mask.sum()),
                "n_drivers": len(drivers),
                "expected_value": expected,
                "top_feature": str(mean_abs.index[0]),
                "top_share": share,
                "dominant": bool(share > DOMINANCE_SHARE),
                "repdte_min": str(rows["repdte"].min().date()),
                "repdte_max": str(rows["repdte"].max().date()),
                "mean_abs_shap": {k: float(v) for k, v in mean_abs.items()},
            }
        )
        run.finish()
        log.info(
            "explained %s with the %d booster: %d rows -> %s", year, model_year, len(rows), path
        )
    return drivers, mean_abs


def rebuild_drivers_table(settings: Settings, models: tuple[str, ...] = (MODEL,)) -> pd.DataFrame:
    """Fold the ``data/drivers/*_<model>.parquet`` files into the ``drivers`` table.

    Only the files of ``models`` (the production booster by default: CONTRACT section 16
    keeps ``gbdt_mono`` rows only, and a second family would double the table past five
    million rows) are read, in sorted name order; :func:`write_table` sorts by the table
    key, so the table is a deterministic function of the per-year files and a re-explained
    year replaces its earlier rows, nothing is ever appended in place.
    """
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import read_table, table_path, write_table

    files = sorted(f for m in models for f in drivers_dir(settings).glob(f"*_{m}.parquet"))
    if not files:
        raise FileNotFoundError(f"no {models} driver files under {drivers_dir(settings)}")
    table = pd.concat([pd.read_parquet(f)[list(COLUMNS)] for f in files], ignore_index=True)
    table = table.astype({"repdte": "datetime64[ns]", "model": "str", "feature": "str"})
    write_table(table, TABLE, key=TABLE_KEY, settings=settings)
    replace_table(TABLE, table_path(settings, TABLE), settings)
    log.info("drivers rebuilt from %d files: %d rows", len(files), len(table))
    return read_table(TABLE, settings)


def explained_years(settings: Settings, model: str = MODEL) -> list[str]:
    """Years (and ``'production'``) with a ``model`` driver file, test years first in order."""
    stems = [
        p.name.removesuffix(f"_{model}.parquet")
        for p in drivers_dir(settings).glob(f"*_{model}.parquet")
    ]
    years = sorted(s for s in stems if s.isdigit())
    return years + [s for s in stems if s == PRODUCTION]


def mean_abs_by_year(settings: Settings, model: str = MODEL) -> pd.DataFrame:
    """``DataFrame[feature x year]`` of mean |SHAP| read back from the ``explain`` run records."""
    from bankcanary import tracking
    from bankcanary.evaluation.walkforward import FEATURE_VERSION

    columns: dict[str, pd.Series] = {}
    for year in explained_years(settings, model):
        model_year = int(year) if year != PRODUCTION else None
        if model_year is None:
            frame = pd.read_parquet(drivers_path(settings, year, model), columns=["model_year"])
            model_year = int(frame["model_year"].iloc[0]) if len(frame) else 0
        config = {
            "model": model,
            "horizon": HORIZON,
            "year": str(year),
            "model_year": model_year,
            "features_version": FEATURE_VERSION,
        }
        metrics = tracking.find_metrics(RUN_NAME, config, settings)
        if metrics is None:
            log.warning("no explain run record for %s; skipped in the summary", year)
            continue
        columns[str(year)] = pd.Series(metrics["mean_abs_shap"], dtype=float)
    if not columns:
        raise FileNotFoundError("no explain run records found; run `bankcanary explain` first")
    return pd.DataFrame(columns).fillna(0.0)


def pooled_summary(
    by_year: pd.DataFrame, weights: dict[str, int] | None = None, normalise: bool = True
) -> pd.DataFrame:
    """Per-feature importance pooled over the test years (weighted when given) plus share.

    Columns: ``feature, mean_abs_shap, share, rank``. ``mean_abs_shap`` is the plain
    (weighted) average of the yearly mean |SHAP| values. ``share`` is the pooled
    importance normalised per year: each year's mean |SHAP| is first divided by that
    year's total, so every year contributes one unit whatever the log-odds scale of its
    booster (the four-failure 2020 fit otherwise outweighs the other sixteen years
    together), and the shares are then averaged over years; ``normalise=False`` derives
    ``share`` from ``mean_abs_shap`` instead. Rows are sorted by ``share`` descending,
    ties by feature name; ``share`` is the quantity spec rule 6.6 asks to watch for a
    suspiciously dominant feature.
    """
    years = [c for c in by_year.columns if c != PRODUCTION]
    w = pd.Series({y: float((weights or {}).get(y, 1)) for y in years})
    block = by_year[years].astype(float)
    pooled = (block * w).sum(axis=1) / w.sum()
    if normalise:
        totals = block.sum(axis=0).replace(0.0, np.nan)
        share = ((block / totals).fillna(0.0) * w).sum(axis=1) / w.sum()
    else:
        total = float(pooled.sum())
        share = pooled / total if total > 0 else pooled * 0.0
    table = pd.DataFrame({"mean_abs_shap": pooled, "share": share}).rename_axis("feature")
    table = table.reset_index().sort_values(
        ["share", "mean_abs_shap", "feature"], ascending=[False, False, True], kind="mergesort"
    )
    return table.assign(rank=np.arange(1, len(table) + 1)).reset_index(drop=True)


def dominance_check(summary: pd.DataFrame, threshold: float = DOMINANCE_SHARE) -> list[str]:
    """Feature names whose share of the total mean |SHAP| exceeds ``threshold`` (rule 6.6)."""
    return summary.loc[summary["share"] > threshold, "feature"].tolist()


def figure_path(settings: Settings) -> Path:
    return Path(settings.reports_dir) / "figures" / "shap_summary_latest.png"


def report_path(settings: Settings) -> Path:
    return Path(settings.reports_dir) / "shap_summary.md"


def beeswarm(values: np.ndarray, clipped: np.ndarray, features: list[str], path, top: int = 15):
    """Beeswarm of the ``top`` features by mean |SHAP| for one block of rows; deterministic PNG.

    Every row's contribution is a dot, coloured by the (winsorised) feature value rank,
    so the picture shows both how much a feature moves the log-odds and in which
    direction high values push. Rows are subsampled with a fixed seed when there are
    more than 4,000 so the file stays small.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bankcanary.evaluation.plots import _METADATA

    values, clipped = np.asarray(values, dtype=float), np.asarray(clipped, dtype=float)
    rng = np.random.default_rng(0)
    if len(values) > 4000:
        keep = np.sort(rng.choice(len(values), 4000, replace=False))
        values, clipped = values[keep], clipped[keep]
    order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")[:top][::-1]
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(order) + 1.2))
    for i, j in enumerate(order):
        x = values[:, j]
        v = clipped[:, j]
        rank = pd.Series(v).rank(pct=True).fillna(0.5).to_numpy()
        jitter = rng.uniform(-0.3, 0.3, size=len(x))
        ax.scatter(x, i + jitter, c=rank, cmap="coolwarm", s=4, alpha=0.6, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([features[j] for j in order], fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("SHAP value (log-odds contribution); colour = feature value rank, red high")
    ax.set_title("SHAP beeswarm, latest walk-forward booster", fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110, format="png", metadata=_METADATA)
    plt.close(fig)
    return path


def write_summary(frame: pd.DataFrame, settings: Settings, path=None, model: str = MODEL) -> Path:
    """Write ``reports/shap_summary.md`` and the latest beeswarm from the stored run records.

    The tables are the mean |SHAP| per feature pooled over every explained test year and
    for the crisis year 2009 against the calm year 2023, the rule 6.6 dominance check
    (any feature above :data:`DOMINANCE_SHARE` of the total is flagged), and the top
    drivers of the production rows. The beeswarm recomputes the SHAP values of the most
    recent test year (a few seconds) rather than storing the full matrices.
    """
    from bankcanary.evaluation.walkforward import load_year
    from bankcanary.models.train import _md_table

    path = Path(path) if path is not None else report_path(settings)
    by_year = mean_abs_by_year(settings, model)
    years = [c for c in by_year.columns if c != PRODUCTION]
    summary = pooled_summary(by_year)
    flagged = dominance_check(summary)
    latest = max(years)
    mask, model_year = select_rows(frame, int(latest), HORIZON)
    pipeline, features, _ = load_year(settings, model_year, model, HORIZON)
    values, _, clipped = shap_matrix(pipeline, frame.loc[mask, list(features)])
    fig = beeswarm(values, clipped, features, figure_path(settings))
    compare_years = [y for y in ("2009", "2023") if y in by_year.columns]
    compare = summary[["feature", "mean_abs_shap", "share"]].head(20).copy()
    for y in compare_years:
        compare[f"mean_abs_{y}"] = compare["feature"].map(by_year[y]).to_numpy()
    top3 = summary.head(3)
    lines = [
        "# SHAP driver summary (walk-forward gradient boosters)",
        "",
        f"`shap.TreeExplainer` on the per-year `{model}` walk-forward models "
        f"({years[0]}-{latest}, each explaining its own test rows) and on the production "
        f"model (the {model_year} booster scoring every quarter after {latest}). Values are "
        "log-odds contributions; `mean_abs_shap` is the mean |SHAP| over a year's rows, "
        "pooled as the plain average over test years, and `share` is the per-year "
        "normalised importance: each year's mean |SHAP| divided by that year's total, then "
        "averaged over years, so a booster on a larger log-odds scale (the four-failure "
        "2020 fit) counts as one year like every other. The `drivers` table keeps the five "
        "largest positive and five largest negative contributions per bank-quarter.",
        "",
        "## Mean |SHAP| per feature, pooled and 2009 versus 2023 (top 20)",
        "",
        _md_table(compare),
        "",
        *_shift_lines(by_year, compare_years),
        "## Feature-importance smoke test (spec rule 6.6)",
        "",
        f"Largest per-year normalised share: `{top3.iloc[0]['feature']}` at "
        f"{top3.iloc[0]['share']:.1%}; the top three together carry "
        f"{top3['share'].sum():.1%}. Threshold {DOMINANCE_SHARE:.0%}: "
        + (
            f"**flagged** {', '.join(f'`{f}`' for f in flagged)}; check the feature for a "
            "leakage artifact before trusting the model."
            if flagged
            else "no feature is flagged; importance is spread across capital, asset-quality, "
            "earnings and sensitivity ratios, with no single field acting as a failure marker."
        ),
        "",
        "## Per-year top feature",
        "",
        _md_table(_top_by_year(by_year)),
        "",
        f"![SHAP beeswarm, latest booster]({fig.relative_to(path.parent).as_posix()})",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def _top_by_year(by_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in by_year.columns:
        s = by_year[col].sort_values(ascending=False, kind="mergesort")
        total = float(s.sum())
        rows.append(
            {
                "year": col,
                "top_feature": s.index[0],
                "top_share": float(s.iloc[0] / total) if total > 0 else 0.0,
                "second_feature": s.index[1] if len(s) > 1 else "",
                "third_feature": s.index[2] if len(s) > 2 else "",
            }
        )
    return pd.DataFrame(rows)


def _shift_lines(by_year: pd.DataFrame, years: list[str]) -> list[str]:
    """One sentence on which features gained and lost weight between two explained years."""
    if len(years) != 2:
        return []
    a, b = years
    shift = (by_year[b] - by_year[a]).sort_values(kind="mergesort")
    fell = ", ".join(f"`{f}` ({v:+.3f})" for f, v in shift.head(3).items())
    rose = ", ".join(f"`{f}` ({v:+.3f})" for f, v in shift.tail(3)[::-1].items())
    return [
        f"Shift in mean |SHAP| from {a} to {b}: rose most for {rose}; fell most for {fell}. "
        "The crisis-year booster leans on credit quality and the housing cycle, the recent "
        "one on rate sensitivity (unrealised losses, the funds-rate change) and capital.",
        "",
    ]
