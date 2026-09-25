"""Data-quality report: what the warehouse contains and where it is thin.

Every section is one DuckDB query over the Parquet tables (``financials_raw``, ``panel``,
``labels``, ``failures``), so the report rebuilds in seconds and is deterministic: the
same tables give the same markdown byte for byte. The sections answer the questions a
modeller asks before trusting Call Report data: how many banks report each quarter, which
fields go missing when the forms change (FFIEC 051 in 2017, the community bank leverage
ratio in 2020, CECL from 2020 onward), whether every failed bank has a report to learn
from, whether raw dollar fields carry sign or scale surprises, and how many labelled
rows each year contributes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from bankcanary.config import Settings
from bankcanary.fields import ATTRIBUTE_GROUPS, DEFAULT_FIELD_MAP_PATH, load_field_map
from bankcanary.storage.parquet import table_path

log = logging.getLogger(__name__)

#: Raw dollar fields whose distribution is summarised in the outlier section.
OUTLIER_FIELDS = ("asset", "eq", "lnlsgr", "nclnls", "netinc", "dep", "bro", "ore", "lnatres")
#: Tables the report reads; each is exposed as a DuckDB view over its Parquet file.
REPORT_TABLES = ("financials_raw", "panel", "labels", "failures")
#: A null share that jumps by at least this much between consecutive years is "structural".
STRUCTURAL_JUMP = 0.3


def open_warehouse(settings: Settings) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one view per Parquet table that exists under ``data_dir``.

    Reading the Parquet files directly (rather than ``warehouse.duckdb``) keeps the report
    tied to the canonical store and lets tests point it at a tiny ``tmp_path`` warehouse.
    """
    con = duckdb.connect()
    for name in REPORT_TABLES:
        path = table_path(settings, name)
        if path.exists():
            literal = str(path).replace("'", "''")
            con.execute(f"CREATE VIEW \"{name}\" AS SELECT * FROM read_parquet('{literal}')")
    return con


def _has_view(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and row[0])


def rows_per_quarter(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Banks reporting each quarter, one row per year with columns Q1..Q4 (0 = no data)."""
    df = con.execute(
        "SELECT year(repdte) AS year, quarter(repdte) AS q, count(*) AS n "
        "FROM financials_raw GROUP BY 1, 2 ORDER BY 1, 2"
    ).df()
    wide = df.pivot(index="year", columns="q", values="n").reindex(columns=[1, 2, 3, 4])
    wide = wide.fillna(0).astype(int)
    wide.columns = [f"Q{c}" for c in wide.columns]
    return wide.reset_index()


def field_columns(specs=None) -> list[str]:
    """Lower-case columns of every financial code in ``config/fields.yaml`` (no identity)."""
    specs = load_field_map() if specs is None else specs
    return [s.column for s in specs if s.group not in ATTRIBUTE_GROUPS]


def _present(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> list[str]:
    have = {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()}
    return [c for c in columns if c in have]


def missingness_by_year(
    con: duckdb.DuckDBPyConnection, columns: list[str] | None = None
) -> pd.DataFrame:
    """Share of null values per field (rows) and report year (columns) in ``financials_raw``.

    A share near 1.0 for a stretch of years means the item did not exist on the form yet
    (or was retired); a share between means only some filers report it (e.g. items that
    the short FFIEC 051 form omits for small banks).
    """
    columns = _present(con, "financials_raw", columns or field_columns())
    exprs = ", ".join(f'count("{c}") AS "{c}"' for c in columns)
    df = con.execute(
        f"SELECT year(repdte) AS year, count(*) AS n_rows, {exprs} "
        "FROM financials_raw GROUP BY 1 ORDER BY 1"
    ).df()
    years = df["year"].astype(int).tolist()
    shares = 1.0 - df[columns].to_numpy(dtype=float).T / df["n_rows"].to_numpy(dtype=float)
    return pd.DataFrame(shares, index=pd.Index(columns, name="field"), columns=years)


def structural_changes(miss: pd.DataFrame, jump: float = STRUCTURAL_JUMP) -> pd.DataFrame:
    """Fields whose null share moves by at least ``jump`` between consecutive years.

    Columns: ``field, year, before, after, change`` where ``change`` is ``appears`` (nulls
    fall: the item was added to the form) or ``disappears`` (nulls rise: retired or
    replaced, as with the CECL allowance items from 2020).
    """
    rows: list[dict[str, Any]] = []
    years = list(miss.columns)
    for field, series in miss.iterrows():
        for prev, year in zip(years[:-1], years[1:]):
            before, after = float(series[prev]), float(series[year])
            if abs(after - before) >= jump:
                kind = "appears" if after < before else "disappears"
                rows.append(
                    {"field": field, "year": year, "before": before, "after": after, "change": kind}
                )
    return pd.DataFrame(rows, columns=["field", "year", "before", "after", "change"])


def first_available(
    con: duckdb.DuckDBPyConnection, columns: list[str] | None = None
) -> dict[str, str | None]:
    """First quarter end with a non-null value per field, as ISO dates (None if never)."""
    columns = _present(con, "financials_raw", columns or field_columns())
    exprs = ", ".join(f'min(repdte) FILTER (WHERE "{c}" IS NOT NULL) AS "{c}"' for c in columns)
    row = con.execute(f"SELECT {exprs} FROM financials_raw").fetchone()
    out: dict[str, str | None] = {}
    for col, value in zip(columns, row):
        missing = value is None or pd.isna(value)
        out[col] = None if missing else pd.Timestamp(value).date().isoformat()
    return out


def failed_bank_matching(con: duckdb.DuckDBPyConnection, start: str = "2001-01-01") -> dict:
    """Failures (``restype = 'FAILURE'``) from ``start`` on, and whether each has a prior report.

    A failure is *matched* when the panel holds at least one row for that cert dated
    before the failure; only matched failures can ever become positive labels. Returns
    ``failures, matched, share`` and ``unmatched`` (DataFrame ``cert, name, fail_date``).
    """
    df = con.execute(
        """
        WITH f AS (
            SELECT cert, min(fail_date) AS fail_date, arg_min(name, fail_date) AS name
            FROM failures WHERE restype = 'FAILURE' AND fail_date >= ?::DATE GROUP BY cert
        )
        SELECT f.cert, f.name, f.fail_date::DATE AS fail_date,
               EXISTS (SELECT 1 FROM panel p WHERE p.cert = f.cert AND p.repdte < f.fail_date)
                   AS matched
        FROM f ORDER BY f.fail_date, f.cert
        """,
        [start],
    ).df()
    unmatched = df.loc[~df["matched"], ["cert", "name", "fail_date"]].reset_index(drop=True)
    n, m = len(df), int(df["matched"].sum())
    share = (m / n) if n else float("nan")
    return {"failures": n, "matched": m, "share": share, "unmatched": unmatched}


def outlier_summary(
    con: duckdb.DuckDBPyConnection, fields: tuple[str, ...] | list[str] = OUTLIER_FIELDS
) -> pd.DataFrame:
    """Tails of key raw dollar fields: ``p0_5, p50, p99_5, negatives, nulls`` per field.

    Negative equity or a negative allowance is real (insolvent banks report it); negative
    total assets or deposits would be a data error. The 0.5/99.5 percentiles are the
    winsorisation points the modelling pipeline later fits on the training fold.
    """
    fields = _present(con, "financials_raw", list(fields))
    rows = []
    for f in fields:
        p05, p50, p995, neg, nulls = con.execute(
            f'SELECT quantile_cont("{f}", 0.005), quantile_cont("{f}", 0.5), '
            f'quantile_cont("{f}", 0.995), count(*) FILTER (WHERE "{f}" < 0), '
            f'count(*) FILTER (WHERE "{f}" IS NULL) FROM financials_raw'
        ).fetchone()
        rows.append(
            {"field": f, "p0_5": p05, "p50": p50, "p99_5": p995, "negatives": neg, "nulls": nulls}
        )
    return pd.DataFrame(rows, columns=["field", "p0_5", "p50", "p99_5", "negatives", "nulls"])


def label_summary_by_year(con: duckdb.DuckDBPyConnection, horizon: int = 4) -> pd.DataFrame:
    """Per report year: rows, usable rows, positives and censored rows for one horizon.

    ``usable`` = label complete and not dropped for a failure before the availability
    date, i.e. the rows a model may train or test on; positives and censored counts are
    taken among usable rows.
    """
    h = int(horizon)
    return con.execute(
        f"""
        SELECT year(repdte) AS year, count(*) AS rows,
               count(*) FILTER (WHERE label_complete_{h}q AND NOT dropped_failed_before_avail)
                   AS usable,
               count(*) FILTER (WHERE y_{h}q = 1 AND label_complete_{h}q
                                AND NOT dropped_failed_before_avail) AS positives,
               count(*) FILTER (WHERE censored_in_window_{h}q AND label_complete_{h}q
                                AND NOT dropped_failed_before_avail) AS censored,
               count(*) FILTER (WHERE dropped_failed_before_avail) AS dropped
        FROM labels GROUP BY 1 ORDER BY 1
        """
    ).df()


FIELD_MAP_HEADER = (
    "# Verified FDIC /financials field map. Generated by scripts/build_field_map.py;\n"
    "# edit the candidate catalog in src/bankcanary/fields.py, not this file.\n"
    "# first_available is filled by the data-quality step.\n"
)


def update_first_available(
    dates: dict[str, str | None], path: Path = DEFAULT_FIELD_MAP_PATH
) -> int:
    """Write ``first_available`` into ``config/fields.yaml`` for the columns in ``dates``.

    Only that value changes: field order, every other key and the header comment are kept,
    and the file is re-serialised exactly as ``fields.write_field_map`` does. Identity and
    attribute columns absent from ``dates`` are left untouched. Returns the number of
    fields whose value changed.
    """
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = 0
    for row in doc["fields"]:
        col = row["column"]
        if col in dates and row.get("first_available") != dates[col]:
            row["first_available"] = dates[col]
            changed += 1
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(FIELD_MAP_HEADER + body, encoding="utf-8")
    log.info("fields.yaml: first_available updated for %d field(s)", changed)
    return changed


#: Integer columns printed without thousands separators (identifiers, not quantities).
PLAIN_INT_COLUMNS = frozenset({"year", "cert"})


def _fmt(value: Any, digits: int, column: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value) if column in PLAIN_INT_COLUMNS else f"{value:,}"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (deterministic text)."""
    frame = df.reset_index() if index else df
    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in frame.itertuples(index=False):
        cells = (_fmt(_py(v), digits, c) for c, v in zip(header, row))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _py(value: Any) -> Any:
    """Convert numpy scalars to Python ones so ``_fmt`` can branch on the type."""
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def render_report(sections: dict[str, Any]) -> str:
    """Assemble the markdown from the section results of :func:`collect_sections`."""
    miss: pd.DataFrame = sections["missingness"]
    changes: pd.DataFrame = sections["structural_changes"]
    match = sections["matching"]
    parts = [
        "# Data-quality report",
        "",
        "Generated by `bankcanary dq-report` from the Parquet tables under `data/parquet/`; "
        "re-running on the same tables reproduces this file exactly. Dollar amounts are in "
        "thousands. Null shares are fractions of the rows reported in that year.",
        "",
        f"Latest report quarter: **{sections['latest_repdte']}**. "
        f"Rows in `financials_raw`: **{sections['rows']:,}** across "
        f"**{sections['quarters']:,}** quarters and **{sections['banks']:,}** banks.",
        "",
        "## 1. Banks reporting per quarter",
        "",
        md_table(sections["rows_per_quarter"]),
        "",
        "## 2. Missingness by field and year",
        "",
        "Structural changes (null share moves by at least "
        f"{STRUCTURAL_JUMP:.0%} between consecutive years):",
        "",
        md_table(changes) if len(changes) else "_none_",
        "",
        "Fields with at least 5% nulls in the latest full year "
        f"({sections['last_full_year']}): "
        + (", ".join(f"`{f}`" for f in sections["partially_missing"]) or "_none_")
        + ".",
        "",
        "Null share per field (rows) and report year (columns):",
        "",
        md_table(miss, digits=3, index=True),
        "",
        "## 3. First available quarter per field",
        "",
        "Written back into `config/fields.yaml` as `first_available`. Fields first "
        "reported after the start of the sample:",
        "",
        md_table(sections["late_fields"]) if len(sections["late_fields"]) else "_none_",
        "",
        "## 4. Failed-bank matching",
        "",
        f"Failures (`restype = 'FAILURE'`) from 2001 onward: **{match['failures']:,}**; with at "
        f"least one prior panel row: **{match['matched']:,}** ({match['share']:.1%}).",
        "",
        "Unmatched failures:",
        "",
        md_table(match["unmatched"]) if len(match["unmatched"]) else "_none_",
        "",
        "## 5. Outlier summary (raw fields, thousands of dollars)",
        "",
        md_table(sections["outliers"], digits=1),
        "",
        "## 6. Labels per report year",
        "",
    ]
    for h, table in sections["labels"].items():
        parts += [f"Horizon {h} quarters (`y_{h}q`):", "", md_table(table), ""]
    return "\n".join(parts).rstrip() + "\n"


def collect_sections(con: duckdb.DuckDBPyConnection, horizons: list[int]) -> dict[str, Any]:
    """Run every section query and return the pieces :func:`render_report` needs."""
    specs = load_field_map()
    columns = field_columns(specs)
    rows, quarters, banks, latest = con.execute(
        "SELECT count(*), count(DISTINCT repdte), count(DISTINCT cert), max(repdte)::DATE "
        "FROM financials_raw"
    ).fetchone()
    miss = missingness_by_year(con, columns)
    per_quarter = rows_per_quarter(con)
    full_years = per_quarter.loc[(per_quarter[["Q1", "Q2", "Q3", "Q4"]] > 0).all(axis=1), "year"]
    last_full = int(full_years.max()) if len(full_years) else int(miss.columns.max())
    partial = [f for f, v in miss[last_full].items() if v >= 0.05] if last_full in miss else []
    dates = first_available(con, columns)
    start = min(d for d in dates.values() if d) if any(dates.values()) else None
    late = pd.DataFrame(
        [{"field": f, "first_available": d} for f, d in dates.items() if d != start],
        columns=["field", "first_available"],
    )
    labels = {}
    if _has_view(con, "labels"):
        labels = {h: label_summary_by_year(con, h) for h in horizons}
    return {
        "rows": int(rows),
        "quarters": int(quarters),
        "banks": int(banks),
        "latest_repdte": str(latest),
        "rows_per_quarter": per_quarter,
        "missingness": miss,
        "structural_changes": structural_changes(miss),
        "last_full_year": last_full,
        "partially_missing": partial,
        "first_available": dates,
        "late_fields": late,
        "matching": failed_bank_matching(con),
        "outliers": outlier_summary(con),
        "labels": labels,
    }


def build_report(
    settings: Settings, field_map_path: Path = DEFAULT_FIELD_MAP_PATH
) -> tuple[Path, dict[str, Any]]:
    """Generate ``reports/data_quality.md`` and update ``first_available`` in the field map.

    Returns the report path and the section results (for the CLI summary and tests).
    """
    con = open_warehouse(settings)
    try:
        sections = collect_sections(con, list(settings.horizons_quarters))
    finally:
        con.close()
    sections["fields_updated"] = update_first_available(sections["first_available"], field_map_path)
    out = settings.reports_dir / "data_quality.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(sections), encoding="utf-8")
    log.info("wrote %s", out)
    return out, sections
