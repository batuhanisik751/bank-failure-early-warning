"""The Prototype 2 metrics suite over the walk-forward scores (spec 8.3, CONTRACT section 13).

Everything here is computed from the ``walkforward_scores`` table, so it can be
regenerated at any time without refitting a model: per-year and pooled ranking metrics
with cluster-bootstrap confidence intervals (banks resampled, not bank-quarters), the
Brier score of the raw and the isotonic-calibrated scores, reliability curves pooled over
every test year by score decile, and the lead time with which each failed bank first
entered the top 2 percent of its quarter's ranking. ``write_metrics_report`` renders
``reports/walkforward.md`` and the ``reliability_<model>.png`` / ``lead_time_<model>.png``
figures, and logs one ``metrics`` run per model and horizon so the headline numbers are
readable from ``runs/`` without parsing the report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bankcanary.config import Settings
from bankcanary.evaluation import walkforward as w
from bankcanary.evaluation.metrics import (
    LOW_CONFIDENCE_FAILURES,
    brier,
    cluster_bootstrap_ci,
    evaluate,
    lead_time_quarters,
    lead_time_summary,
)

log = logging.getLogger(__name__)

RUN_NAME = "metrics"
#: The failure-date source: the panel's ``fail_date`` (earliest FAILURE on or after estymd).
FAIL_DATE_TABLE = "panel"
#: The spec's crisis cohort for the lead-time headline.
LEAD_TIME_YEARS: tuple[int, int] = (2009, 2012)
TOP_FRAC = 0.02
N_BINS = 10
#: Lead times at or beyond this many quarters share one histogram bar.
LEAD_TIME_CAP = 20
POOLED = "pooled"


def failure_dates(settings: Settings) -> pd.Series:
    """``cert -> fail_date`` for every bank that failed, read from the panel's two columns."""
    import pyarrow.parquet as pq

    from bankcanary.storage.parquet import table_path

    cols = pq.read_table(table_path(settings, FAIL_DATE_TABLE), columns=["cert", "fail_date"])
    frame = cols.to_pandas().dropna(subset=["fail_date"])
    return frame.groupby("cert")["fail_date"].min().astype("datetime64[ns]")


def _slices(sub: pd.DataFrame):
    """``(label, rows)`` per test year in order, then the pooled slice."""
    for year in sorted(int(y) for y in sub["test_year"].unique()):
        yield str(year), sub[sub["test_year"] == year]
    yield POOLED, sub


def ci_table(
    sub: pd.DataFrame, score_col: str = "score", n_draws: int | None = None
) -> pd.DataFrame:
    """Per-year and pooled PR-AUC / recall@2% with cluster-bootstrap 95 percent intervals.

    One row per test year plus ``pooled``; ``low_confidence`` marks the years holding
    fewer than :data:`LOW_CONFIDENCE_FAILURES` failures, whose interval (and point
    estimate) should not be read as a ranking of the models.
    """
    kwargs = {} if n_draws is None else {"n_draws": int(n_draws)}
    rows = []
    for label, rows_ in _slices(sub):
        y, s, c = rows_["y"], rows_[score_col], rows_["cert"]
        point = evaluate(y, s, tie_breaker=c)
        ci = cluster_bootstrap_ci(y, s, c, tie_breaker=c, k_frac=TOP_FRAC, **kwargs)
        rows.append(
            {
                "year": label,
                "n": point["n"],
                "n_failures": point["n_failures"],
                "pr_auc": point["pr_auc"],
                "pr_auc_low": ci["pr_auc_ci"][0],
                "pr_auc_high": ci["pr_auc_ci"][1],
                "recall_at_2pct": point["recall_at_2pct"],
                "recall_at_2pct_low": ci["recall_at_2pct_ci"][0],
                "recall_at_2pct_high": ci["recall_at_2pct_ci"][1],
                "roc_auc": point["roc_auc"],
                "low_confidence": bool(ci["low_confidence"]),
            }
        )
    return pd.DataFrame(rows)


def brier_table(sub: pd.DataFrame) -> pd.DataFrame:
    """Brier score of the raw and the calibrated score per test year and pooled.

    ``mean_raw`` / ``mean_calibrated`` against ``failure_rate`` show whether the scores
    are on the probability scale at all: a ranking model with the right order but a
    mean ten times the failure rate has a poor Brier score however good its PR-AUC.
    All-NaN calibrated scores (the Texas ratio) give NaN cells.
    """
    rows = []
    for label, r in _slices(sub):
        y = r["y"].to_numpy(dtype=float)
        raw, cal = r["score"].to_numpy(dtype=float), r["score_calibrated"].to_numpy(dtype=float)
        have_cal = ~np.isnan(cal)
        rows.append(
            {
                "year": label,
                "n": int(len(r)),
                "n_failures": int(np.nansum(y)),
                "failure_rate": float(np.nanmean(y)) if len(y) else float("nan"),
                "brier_raw": brier(y, np.nan_to_num(raw, nan=0.0)),
                "brier_calibrated": brier(y[have_cal], cal[have_cal])
                if have_cal.any()
                else float("nan"),
                "mean_raw": float(np.nanmean(raw)) if len(raw) else float("nan"),
                "mean_calibrated": float(np.mean(cal[have_cal]))
                if have_cal.any()
                else float("nan"),
                "low_confidence": int(np.nansum(y)) < LOW_CONFIDENCE_FAILURES,
            }
        )
    return pd.DataFrame(rows)


def reliability_table(sub: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Observed failure rate against mean raw and calibrated score by raw-score decile.

    The bins are deciles of the *raw* score over the pooled rows (rank based, so ties
    and heavy tails do not empty a bin); isotonic calibration is monotone in the raw
    score, so the same bins serve both curves and the two columns can be compared bin
    by bin. A perfectly calibrated score has ``mean_* == observed_rate`` in every bin.
    """
    r = sub.dropna(subset=["score"])
    if r.empty:
        return pd.DataFrame(
            columns=["decile", "n", "n_failures", "mean_raw", "mean_calibrated", "observed_rate"]
        )
    ranks = r["score"].rank(method="first")
    decile = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop") + 1
    grouped = r.assign(decile=decile).groupby("decile", sort=True)
    out = grouped.agg(
        n=("y", "size"),
        n_failures=("y", "sum"),
        mean_raw=("score", "mean"),
        mean_calibrated=("score_calibrated", "mean"),
        observed_rate=("y", "mean"),
    ).reset_index()
    return out.astype({"decile": int, "n": int, "n_failures": int})


def lead_time_tables(sub: pd.DataFrame, fail_dates: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Per-failed-bank lead times and their summaries for the crisis cohort and for all.

    ``fail_dates`` maps ``cert`` to the failure date; banks without a scored quarter
    before their failure are not in the table (a bank failing in the first test year
    can only be flagged in its own year, so 2008 failures have short lead times by
    construction, which is why the spec's headline is the 2009-2012 cohort).
    """
    d = sub.assign(fail_date=sub["cert"].map(fail_dates))
    lead = lead_time_quarters(d, "score", "cert", "repdte", "fail_date", top_frac=TOP_FRAC)
    summaries = {
        "crisis": lead_time_summary(lead, fail_years=LEAD_TIME_YEARS),
        "all": lead_time_summary(lead),
    }
    return lead, summaries


def reliability_figure(rel: pd.DataFrame, model: str, path) -> Path:
    """Reliability curve: observed failure rate against the mean score per raw-score decile.

    Both axes are logarithmic because a 0.5 percent base rate puts nine of ten deciles
    below one percent; the dashed diagonal is perfect calibration. Points above the
    diagonal are under-predicted deciles, points below it over-predicted ones.
    """
    import matplotlib.pyplot as plt

    from bankcanary.evaluation.plots import _FIGSIZE, _save

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    floor = 1e-5
    if not rel.empty:
        obs = np.clip(rel["observed_rate"].to_numpy(dtype=float), floor, None)
        for col, label, marker in (("mean_raw", "raw", "o"), ("mean_calibrated", "isotonic", "s")):
            x = rel[col].to_numpy(dtype=float)
            if np.isnan(x).all():
                continue
            ax.plot(np.clip(x, floor, None), obs, marker=marker, linewidth=1, label=label)
        lo, hi = floor, 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="grey", label="perfect")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no scores", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Mean score in decile")
    ax.set_ylabel("Observed failure rate")
    ax.set_title(f"Reliability by score decile: {model}", fontsize=10)
    return _save(fig, path)


def lead_time_figure(lead: pd.DataFrame, model: str, path) -> Path:
    """Histogram of quarters between a failed bank's first top-2% flag and its failure.

    Lead times of :data:`LEAD_TIME_CAP` quarters or more share one bar, the right-most
    bar counts the failed banks the ranking never flagged before they failed, and the
    dotted line marks the spec's two-quarter target.
    """
    import matplotlib.pyplot as plt

    from bankcanary.evaluation.plots import _FIGSIZE, _save

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    lt = lead["lead_time_quarters"].astype(float) if not lead.empty else pd.Series(dtype=float)
    flagged = lt.dropna().astype(int)
    if len(lt):
        cap = LEAD_TIME_CAP
        counts = flagged.clip(upper=cap).value_counts().reindex(range(0, cap + 1), fill_value=0)
        never = int(lt.isna().sum())
        ax.bar(counts.index, counts.to_numpy(), width=0.8, label=f"flagged ({len(flagged)})")
        ax.bar([cap + 2], [never], width=0.8, color="lightgrey", label=f"never ({never})")
        ax.axvline(1.5, linestyle=":", linewidth=1, color="black")
        ticks = list(range(0, cap, 4)) + [cap, cap + 2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks[:-2]] + [f"{cap}+", "never"], fontsize=8)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no failed banks", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Quarters before failure at first top-2% flag")
    ax.set_ylabel("Failed banks")
    ax.set_title(f"Lead time: {model}", fontsize=10)
    return _save(fig, path)


def model_suite(
    sub: pd.DataFrame, fail_dates: pd.Series | None = None, n_draws: int | None = None
) -> dict:
    """Every table of the suite for one (model, horizon) slice of ``walkforward_scores``.

    Keys: ``ci`` (per-year + pooled ranking metrics with intervals), ``brier``,
    ``reliability`` (only when a calibrated score exists), ``lead`` and ``lead_summary``
    (only when ``fail_dates`` is given).
    """
    suite = {"ci": ci_table(sub, n_draws=n_draws), "brier": brier_table(sub)}
    if sub["score_calibrated"].notna().any():
        suite["reliability"] = reliability_table(sub)
    if fail_dates is not None:
        suite["lead"], suite["lead_summary"] = lead_time_tables(sub, fail_dates)
    return suite


def log_metrics_run(settings: Settings, model: str, horizon: int, suite: dict) -> Path:
    """Write ``runs/metrics/<id>/`` with the pooled headline numbers of one model."""
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED

    ci, br = suite["ci"], suite["brier"]
    years = [int(y) for y in ci["year"] if y != POOLED]
    config = {
        "model": model,
        "horizon": int(horizon),
        "features_version": w.FEATURE_VERSION,
        "test_years": [min(years), max(years)] if years else [],
        "n_draws": int(BOOTSTRAP_DRAWS),
        "seed": int(BOOTSTRAP_SEED),
        "top_frac": TOP_FRAC,
        "lead_time_years": list(LEAD_TIME_YEARS),
    }
    pooled_ci = ci[ci["year"] == POOLED].iloc[0]
    pooled_br = br[br["year"] == POOLED].iloc[0]
    metrics = {
        k: (None if pd.isna(v) else float(v))
        for k, v in {**pooled_ci.drop("year"), **pooled_br.drop("year")}.items()
        if k not in ("low_confidence",)
    }
    metrics["n"], metrics["n_failures"] = int(pooled_ci["n"]), int(pooled_ci["n_failures"])
    metrics["low_confidence_years"] = [
        int(y) for y, f in zip(ci["year"], ci["low_confidence"]) if f and y != POOLED
    ]
    for cohort, summary in suite.get("lead_summary", {}).items():
        for key in (
            "n_failed",
            "n_flagged",
            "share_flagged",
            "median_lead_time_quarters",
            "share_flagged_ahead",
        ):
            metrics[f"lead_{cohort}_{key}"] = summary[key]
    run = tracking.start_run(RUN_NAME, config, settings)
    run.log_metrics(metrics)
    return run.finish()


def _cell(value) -> str:
    return w._cell(value)


def _ci_cell(low, high) -> str:
    if pd.isna(low) or pd.isna(high):
        return "n/a"
    return f"[{float(low):.3f}, {float(high):.3f}]"


def _flag(value) -> str:
    return "yes" if bool(value) else ""


def _year_rows(ci: pd.DataFrame, br: pd.DataFrame) -> list[str]:
    head = (
        "| year | n | failures | pr_auc | pr_auc 95% CI | recall@2% | recall@2% 95% CI | "
        "roc_auc | brier raw | brier calibrated | low confidence |"
    )
    lines = [head, "|---" * 11 + "|"]
    merged = ci.merge(br[["year", "brier_raw", "brier_calibrated"]], on="year")
    for _, r in merged.iterrows():
        lines.append(
            f"| {r['year']} | {int(r['n'])} | {int(r['n_failures'])} | {_cell(r['pr_auc'])} | "
            f"{_ci_cell(r['pr_auc_low'], r['pr_auc_high'])} | {_cell(r['recall_at_2pct'])} | "
            f"{_ci_cell(r['recall_at_2pct_low'], r['recall_at_2pct_high'])} | "
            f"{_cell(r['roc_auc'])} | {_cell(r['brier_raw'])} | {_cell(r['brier_calibrated'])} | "
            f"{_flag(r['low_confidence'])} |"
        )
    return lines


def _pooled_section(suites: dict[str, dict], horizon: int, reference: dict | None) -> list[str]:
    rows = []
    for model, s in suites.items():
        c, b = s["ci"].iloc[-1], s["brier"].iloc[-1]
        rows.append((model, c, b))
    rows.sort(key=lambda t: -(t[1]["pr_auc"] if pd.notna(t[1]["pr_auc"]) else -1.0))
    lines = [
        "| model | n | failures | pr_auc | pr_auc 95% CI | recall@2% | recall@2% 95% CI | "
        "roc_auc | brier raw | brier calibrated |",
        "|---" * 10 + "|",
    ]
    for model, c, b in rows:
        lines.append(
            f"| {model} | {int(c['n'])} | {int(c['n_failures'])} | {_cell(c['pr_auc'])} | "
            f"{_ci_cell(c['pr_auc_low'], c['pr_auc_high'])} | {_cell(c['recall_at_2pct'])} | "
            f"{_ci_cell(c['recall_at_2pct_low'], c['recall_at_2pct_high'])} | "
            f"{_cell(c['roc_auc'])} | {_cell(b['brier_raw'])} | {_cell(b['brier_calibrated'])} |"
        )
    best, c, _ = rows[0]
    lines += [
        "",
        f"Best pooled PR-AUC at {horizon}q: **{best}** ({_cell(c['pr_auc'])}, 95% CI "
        f"{_ci_cell(c['pr_auc_low'], c['pr_auc_high'])}; recall@2% {_cell(c['recall_at_2pct'])}). "
        "Intervals are percentile intervals from 200 cluster-bootstrap draws that resample "
        "banks (certs) with replacement, so the consecutive quarters of one bank move "
        "together; two models whose intervals overlap are not distinguished by this backtest.",
    ]
    if reference is not None and "logit" in suites:
        wf = suites["logit"]["ci"].iloc[-1]
        lines += [
            "",
            f"Prototype 1 fixed-split logit (test 2010-2013, v1 features): PR-AUC "
            f"{_cell(reference.get('pr_auc'))}, recall@2% {_cell(reference.get('recall_at_2pct'))} "
            f"on {reference.get('n_failures')} failures; not comparable with the pooled rows (one "
            f"training window, a different test period). The walk-forward logit pools to PR-AUC "
            f"{_cell(wf['pr_auc'])} / recall@2% {_cell(wf['recall_at_2pct'])} with one model per "
            "year and hyper-parameters re-selected inside each year's training period.",
        ]
    return lines


def _reliability_rows(rel: pd.DataFrame) -> list[str]:
    lines = [
        "| decile | n | failures | mean raw | mean calibrated | observed rate |",
        "|---" * 6 + "|",
    ]
    for _, r in rel.iterrows():
        lines.append(
            f"| {int(r['decile'])} | {int(r['n'])} | {int(r['n_failures'])} | "
            f"{float(r['mean_raw']):.5f} | {_cell(float(r['mean_calibrated']))} | "
            f"{float(r['observed_rate']):.5f} |"
        )
    return lines


def _lead_time_section(suites: dict[str, dict]) -> list[str]:
    lo, hi = LEAD_TIME_YEARS
    lines = [
        "| model | failures | failed banks | flagged before failure | median lead (flagged) | "
        "median lead (never = 0) | flagged >= 2 quarters ahead | low confidence |",
        "|---" * 8 + "|",
    ]
    for model, s in suites.items():
        for cohort, label in (("crisis", f"{lo}-{hi}"), ("all", "all")):
            m = s["lead_summary"][cohort]
            lines.append(
                f"| {model} | {label} | {m['n_failed']} | {m['n_flagged']} "
                f"({_cell(m['share_flagged'])}) | {_cell(m['median_lead_time_quarters'])} | "
                f"{_cell(m['median_lead_time_all'])} | {_cell(m['share_flagged_ahead'])} | "
                f"{_flag(m['low_confidence'])} |"
            )
    return lines


def write_metrics_report(
    settings: Settings, horizon: int = w.PRIMARY_HORIZON, path: Path | None = None
) -> Path:
    """Render ``reports/walkforward.md`` and the reliability / lead-time figures.

    The requested horizon gets the full suite (pooled and per-year tables with
    intervals and Brier scores, the calibration section with reliability deciles, the
    lead-time section with figures); every other horizon in the table gets its pooled
    and per-year tables. One ``metrics`` run is logged per model and horizon.
    """
    table = w.read_scores_table(settings)
    if table.empty:
        raise ValueError("walkforward_scores is empty; run `bankcanary walkforward` first")
    horizons = sorted(int(h) for h in table["horizon"].unique())
    if int(horizon) not in horizons:
        raise ValueError(f"no walk-forward scores at {horizon}q (have {horizons})")
    fail_dates = failure_dates(settings)
    figures = Path(settings.reports_dir) / "figures"
    lines = _intro_lines(settings)
    for h in [int(horizon)] + [x for x in horizons if x != int(horizon)]:
        slice_h = table[table["horizon"] == h]
        years = sorted(int(y) for y in slice_h["test_year"].unique())
        suites = {}
        for model in [m for m in w.MODELS if m in set(slice_h["model"])]:
            sub = slice_h[slice_h["model"] == model]
            suites[model] = model_suite(sub, fail_dates)
            log_metrics_run(settings, model, h, suites[model])
        lines += [f"## Horizon {h}q (test years {years[0]}-{years[-1]})", "", "### Pooled", ""]
        ref = w.p1_fixed_split_reference(settings) if h == int(horizon) else None
        lines += _pooled_section(suites, h, ref) + ["", "### Per-year results", ""]
        for model, s in suites.items():
            lines += [f"#### {model} ({h}q)", "", *_year_rows(s["ci"], s["brier"]), ""]
        lines.append(
            "Years flagged low confidence hold fewer than "
            f"{LOW_CONFIDENCE_FAILURES} failures; a year with no failure has undefined ranking "
            "metrics and still counts in the pooled row through its non-failing banks."
        )
        lines.append("")
        if h == int(horizon):
            lines += _calibration_lines(suites, h, figures)
            lines += _lead_time_lines(suites, figures)
    out = path or w.report_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    log.info("wrote %s", out)
    return out


def _intro_lines(settings: Settings) -> list[str]:
    lag = settings.availability_lag_days
    return [
        "# Walk-forward backtest",
        "",
        "One model per test year: for test year Y the training rows are every earlier "
        "bank-quarter whose outcome window closed before the first prediction date of Y "
        f"(report 03-31 of Y plus the {lag}-day availability lag, spec rule 6.2), the test rows "
        "are the label-complete reports dated in Y, and hyper-parameters are re-selected per "
        "year on a validation slice inside that year's training period (rule 6.7; "
        "`models/walkforward/<Y>/<model>/tuning.json`). `texas` ranks by the Texas ratio "
        "without a fit; `logit` is the regularised logit on the v2 features; `gbdt` the "
        "gradient booster; `hazard` the one-quarter hazard converted with `1 - (1 - h)^H`. "
        "Pooled rows score every test year as one ranking. `brier raw` is the mean squared "
        "error of the model's own output (n/a for the Texas ranking); `brier calibrated` is "
        "that of the isotonic map fitted per year on the last complete label year inside the "
        "training window (an inner model trained on the years before it scores that slice, "
        "so no test outcome shapes the map). Generated by `bankcanary metrics-report`.",
        "",
    ]


def _calibration_lines(suites: dict[str, dict], horizon: int, figures: Path) -> list[str]:
    lines = [
        "### Calibration",
        "",
        "The map for test year Y is learned on the last complete label year inside Y's "
        "training window from the scores of an inner model fitted on the years before that "
        "slice, so it is applied to a full-window model whose score scale it never saw; where "
        "the two scales differ (a crisis year entering the full window, a re-tuned C) the "
        "calibrated probabilities inherit the inner model's plateaus and the Brier score can "
        "worsen. Pooled deciles mix years, so a model whose raw scale drifts between years "
        "shows a non-monotone raw curve even when every single year ranks well.",
        "",
    ]
    for model, s in suites.items():
        if "reliability" not in s:
            continue
        b = s["brier"]
        pooled = b.iloc[-1]
        worse = b[(b["year"] != POOLED) & (b["brier_calibrated"] > b["brier_raw"])]["year"]
        lines += [
            f"#### {model} ({horizon}q)",
            "",
            f"Pooled Brier raw {_cell(pooled['brier_raw'])} against calibrated "
            f"{_cell(pooled['brier_calibrated'])}; pooled mean score {_cell(pooled['mean_raw'])} "
            f"raw and {_cell(pooled['mean_calibrated'])} calibrated against a failure rate of "
            f"{_cell(pooled['failure_rate'])}. Calibration worsens the Brier score in "
            f"{len(worse)} of {len(b) - 1} years"
            + (f" ({', '.join(worse)})." if len(worse) else "."),
            "",
            "| year | failures | failure rate | mean raw | mean calibrated | brier raw | "
            "brier calibrated | low confidence |",
            "|---" * 8 + "|",
        ]
        for _, r in b.iterrows():
            lines.append(
                f"| {r['year']} | {int(r['n_failures'])} | {_cell(r['failure_rate'])} | "
                f"{_cell(r['mean_raw'])} | {_cell(r['mean_calibrated'])} | {_cell(r['brier_raw'])} "
                f"| {_cell(r['brier_calibrated'])} | {_flag(r['low_confidence'])} |"
            )
        fig = reliability_figure(s["reliability"], model, figures / f"reliability_{model}.png")
        lines += ["", "Reliability by raw-score decile, pooled over every test year:", ""]
        lines += _reliability_rows(s["reliability"])
        lines += ["", f"![reliability {model}](figures/{fig.name})", ""]
    return lines


def _lead_time_lines(suites: dict[str, dict], figures: Path) -> list[str]:
    lo, hi = LEAD_TIME_YEARS
    lines = [
        "### Lead time",
        "",
        "For every bank that failed, the number of calendar quarters between the first "
        "report at which it entered the top 2 percent of that quarter's ranking and its "
        f"failure. The {lo}-{hi} cohort is the spec's headline; `all` counts every failure "
        "with at least one scored quarter before it, including the first test year (which "
        "can only be flagged within that year) and failures after the last complete test "
        "year (scored only through it), so its lead times are shortened at both ends. "
        "Never-flagged banks are in every denominator.",
        "",
        *_lead_time_section(suites),
        "",
    ]
    for model, s in suites.items():
        fig = lead_time_figure(s["lead"], model, figures / f"lead_time_{model}.png")
        lines.append(f"![lead time {model}](figures/{fig.name})")
    return lines + [""]
