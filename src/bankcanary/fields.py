"""Verified FDIC field map: which ``/financials`` codes the pipeline pulls and what they mean.

The FDIC BankFind API publishes a data dictionary (``risview_properties.yaml``) with one
entry per code. Codes in the spec are only candidates until they are checked against that
dictionary; this module holds the candidate catalog, the resolution logic and the reader
for the resulting ``config/fields.yaml``. ``scripts/build_field_map.py`` runs the
resolution and writes the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bankcanary.config import PROJECT_ROOT

DEFAULT_FIELD_MAP_PATH = PROJECT_ROOT / "config" / "fields.yaml"
DEFAULT_DICTIONARY_PATH = (
    PROJECT_ROOT / "data" / "raw" / "fdic" / "docs" / "risview_properties.yaml"
)

UNITS = ("thousands_usd", "percent", "ratio", "count", "flag", "text", "date")
GROUPS = (
    "identity",
    "attribute",
    "balance_sheet",
    "loan_mix",
    "asset_quality",
    "income_ytd",
    "capital",
    "funding",
    "securities",
    "other",
)
# Groups whose columns are institution descriptors rather than financial values.
ATTRIBUTE_GROUPS = ("identity", "attribute")


@dataclass(frozen=True)
class FieldSpec:
    """One pulled field: an FDIC code and the meaning the pipeline assigns to it."""

    code: str
    column: str
    description: str
    unit: str
    group: str
    prototype: str
    first_available: str | None = None
    in_dictionary: bool = True


@dataclass(frozen=True)
class Candidate:
    """A code the spec (or the discovery work) proposes, before dictionary verification."""

    code: str
    unit: str
    group: str
    prototype: str = "P1"
    description: str | None = None  # overrides the dictionary title when it is cryptic
    required: bool = True  # a missing required code is reported loudly


@dataclass(frozen=True)
class KeywordSearch:
    """A 'find code' item from the spec: resolved by keyword search over the dictionary."""

    label: str
    pattern: str
    chosen: str
    unit: str
    group: str
    prototype: str = "P1"
    description: str | None = None


@dataclass
class FieldMapResult:
    """Output of :func:`build_field_map`: the resolved fields plus the discovery report."""

    fields: list[FieldSpec]
    not_found: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    search_report: dict[str, list[tuple[str, str]]] = field(default_factory=dict)


def _c(
    code: str, unit: str, group: str, prototype: str = "P1", desc: str | None = None
) -> Candidate:
    return Candidate(code=code, unit=unit, group=group, prototype=prototype, description=desc)


K, PCT, N, F, T, D = "thousands_usd", "percent", "count", "flag", "text", "date"

# Spec section 7.1 candidates plus the codes verified on the live endpoint. Order is the
# column order of ``financials_raw``.
CANDIDATES: tuple[Candidate, ...] = (
    # identity and attributes (as reported on the financials record)
    _c("CERT", N, "identity", desc="FDIC certificate number"),
    _c("REPDTE", D, "identity", desc="Report date (quarter end)"),
    _c("NAME", T, "attribute", desc="Institution name"),
    _c("STALP", T, "attribute", desc="State (two-letter postal code)"),
    _c("STNAME", T, "attribute", desc="State name"),
    _c("CITY", T, "attribute", desc="City"),
    _c("BKCLASS", T, "attribute", desc="Institution class (charter type / regulator)"),
    _c("CB", F, "attribute", desc="Community bank flag (1 = community bank)"),
    _c("RSSDHCR", N, "attribute", desc="Federal Reserve RSSD ID of the top holding company"),
    _c("FED_RSSD", N, "attribute", desc="Federal Reserve RSSD ID of the institution"),
    _c("ESTYMD", D, "attribute", desc="Date the institution was established"),
    # balance sheet (stock items, thousands of dollars)
    _c("ASSET", K, "balance_sheet", desc="Total assets"),
    _c("ASSET2", K, "balance_sheet", desc="Average total assets (two-quarter average)"),
    _c("LIAB", K, "balance_sheet", desc="Total liabilities"),
    _c("DEP", K, "balance_sheet", desc="Total deposits"),
    _c("DEPDOM", K, "balance_sheet", desc="Deposits held in domestic offices"),
    _c("EQ", K, "balance_sheet", desc="Total equity capital"),
    _c("INTAN", K, "balance_sheet", desc="Intangible assets (goodwill + other intangibles)"),
    _c("INTANGW", K, "balance_sheet", desc="Goodwill"),
    _c("LNLSGR", K, "balance_sheet", desc="Gross loans and leases"),
    _c("LNLSNET", K, "balance_sheet", desc="Net loans and leases (after allowance)"),
    _c("LNATRES", K, "balance_sheet", desc="Allowance for loan and lease losses"),
    _c("SC", K, "balance_sheet", desc="Total securities"),
    _c("CHBAL", K, "balance_sheet", desc="Cash and balances due from depository institutions"),
    _c("FREPO", K, "balance_sheet", desc="Fed funds sold and reverse repos"),
    _c("FREPP", K, "balance_sheet", desc="Fed funds purchased and repos"),
    _c("TRADE", K, "balance_sheet", desc="Trading account assets"),
    _c("ORE", K, "balance_sheet", desc="Other real estate owned (foreclosed property)"),
    _c("OTHBFHLB", K, "balance_sheet", desc="FHLB advances"),
    _c("OTHBOR", K, "balance_sheet", desc="Other borrowed money (incl. FHLB advances)"),
    _c("SUBND", K, "balance_sheet", desc="Subordinated notes and debentures"),
    _c("ERNAST", K, "balance_sheet", "P2", desc="Total earning assets"),
    # loan mix (stock items)
    _c("LNRE", K, "loan_mix", desc="Total real estate loans"),
    _c("LNRECONS", K, "loan_mix", desc="Construction and land development loans"),
    _c("LNRENRES", K, "loan_mix", desc="Nonfarm nonresidential real estate loans (total)"),
    _c("LNRENROW", K, "loan_mix", desc="Owner-occupied nonfarm nonresidential loans"),
    _c("LNRENROT", K, "loan_mix", desc="Non-owner-occupied nonfarm nonresidential loans"),
    _c("LNREMULT", K, "loan_mix", desc="Multifamily (5+ units) residential loans"),
    _c("LNRERES", K, "loan_mix", desc="1-4 family residential loans"),
    _c("LNRELOC", K, "loan_mix", desc="Home equity lines of credit (1-4 family revolving)"),
    _c("LNCI", K, "loan_mix", desc="Commercial and industrial loans"),
    _c("LNCON", K, "loan_mix", desc="Consumer loans"),
    _c("LNCRCD", K, "loan_mix", desc="Credit card loans"),
    _c("LNAG", K, "loan_mix", desc="Agricultural production loans"),
    # asset quality (stock items)
    _c(
        "NCLNLS",
        K,
        "asset_quality",
        desc="Noncurrent loans and leases (90+ days past due + nonaccrual)",
    ),
    _c("P3LNLS", K, "asset_quality", desc="Loans and leases 30-89 days past due"),
    _c("P9LNLS", K, "asset_quality", desc="Loans and leases 90+ days past due"),
    _c("NALNLS", K, "asset_quality", desc="Nonaccrual loans and leases"),
    _c("P3ASSET", K, "asset_quality", desc="Total assets 30-89 days past due"),
    _c("P9ASSET", K, "asset_quality", desc="Total assets 90+ days past due"),
    _c("NAASSET", K, "asset_quality", desc="Total nonaccrual assets"),
    _c("NCLNLSR", PCT, "asset_quality", desc="Noncurrent loans / gross loans (published ratio)"),
    # income statement (year-to-date flows; de-accumulated downstream)
    _c("NETINC", K, "income_ytd", desc="Net income, year to date"),
    _c("INTINC", K, "income_ytd", desc="Total interest income, year to date"),
    _c("EINTEXP", K, "income_ytd", desc="Total interest expense, year to date"),
    _c("NIM", K, "income_ytd", desc="Net interest income, year to date"),
    _c("NONII", K, "income_ytd", desc="Total noninterest income, year to date"),
    _c("NONIX", K, "income_ytd", desc="Total noninterest expense, year to date"),
    _c("ELNATR", K, "income_ytd", desc="Provision for credit losses, year to date"),
    _c("NTLNLS", K, "income_ytd", desc="Net loan and lease charge-offs, year to date"),
    _c("DRLNLS", K, "income_ytd", desc="Gross loan and lease charge-offs, year to date"),
    _c("CRLNLS", K, "income_ytd", desc="Loan and lease recoveries, year to date"),
    _c("IGLSEC", K, "income_ytd", desc="Realized securities gains and losses, year to date"),
    _c("ITAX", K, "income_ytd", desc="Applicable income taxes, year to date"),
    _c("NOIJ", K, "income_ytd", desc="Net operating income (adjusted), year to date"),
    _c("EDEP", K, "income_ytd", "P2", desc="Interest expense on deposits, year to date"),
    _c(
        "EDEPDOM", K, "income_ytd", "P2", desc="Interest expense on domestic deposits, year to date"
    ),
    _c("NETINCQ", K, "income_ytd", desc="Net income, quarterly (FDIC-published cross-check)"),
    # capital
    _c("RBC1AAJ", PCT, "capital", desc="Tier 1 leverage ratio (PCA definition)"),
    _c("RBCRWAJ", PCT, "capital", desc="Total risk-based capital ratio (PCA definition)"),
    _c("IDT1RWAJR", PCT, "capital", desc="Tier 1 risk-based capital ratio"),
    _c("RBC1RWAJ", PCT, "capital", desc="Tier 1 risk-based capital ratio (PCA definition)"),
    _c("RBCT1CER", PCT, "capital", desc="Common equity Tier 1 capital ratio (2015+)"),
    _c("RBCT1C", K, "capital", desc="Common equity Tier 1 capital, dollars (2015+)"),
    _c("RBCT2", K, "capital", desc="Tier 2 capital, dollars (PCA definition)"),
    _c("RWAJ", K, "capital", desc="Risk-weighted assets (PCA definition)"),
    _c("CBLRIND", F, "capital", desc="Community bank leverage ratio election (2020+)"),
    # funding
    _c("BRO", K, "funding", desc="Brokered deposits"),
    _c("BROINS", K, "funding", desc="Brokered deposits, fully insured"),
    _c("DEPI", K, "funding", desc="Interest-bearing deposits"),
    _c("DEPNI", K, "funding", desc="Noninterest-bearing deposits"),
    _c("COREDEP", K, "funding", desc="Core deposits (FDIC definition)"),
    _c("TRN", K, "funding", desc="Transaction accounts"),
    _c("DEPINS", K, "funding", "P2", desc="Estimated insured deposits"),
    _c("DEPUNINS", K, "funding", "P2", desc="Estimated uninsured deposits"),
    _c("DEPLGAMT", K, "funding", desc="Deposit accounts above $250,000, amount"),
    _c("DEPSMB", N, "funding", desc="Deposit accounts of $250,000 or less, number"),
    _c(
        "CD3LES",
        K,
        "funding",
        "P2",
        desc="Time deposits >= $250k maturing/repricing in 3 months or less",
    ),
    _c(
        "CD3T12",
        K,
        "funding",
        "P2",
        desc="Time deposits >= $250k maturing/repricing in 3-12 months",
    ),
    _c("CD1T3", K, "funding", "P2", desc="Time deposits >= $250k maturing/repricing in 1-3 years"),
    _c(
        "CDOV3",
        K,
        "funding",
        "P2",
        desc="Time deposits >= $250k maturing/repricing in over 3 years",
    ),
    # securities (Prototype 2 rate-risk items)
    _c("SCAF", K, "securities", "P2", desc="Available-for-sale securities at fair value"),
    _c("SCAA", K, "securities", "P2", desc="Available-for-sale securities at amortized cost"),
    _c("SCHF", K, "securities", "P2", desc="Held-to-maturity securities at fair value"),
    # FDIC-published ratios (cross-checks for the computed features)
    _c("ROA", PCT, "other", desc="Return on assets, annualised YTD (published)"),
    _c("ROAQ", PCT, "other", desc="Return on assets, quarterly annualised (published)"),
    _c("ROE", PCT, "other", desc="Return on equity, annualised YTD (published)"),
    _c("NIMY", PCT, "other", desc="Net interest margin, YTD (published)"),
    _c("NIMYQ", PCT, "other", desc="Net interest margin, quarterly (published)"),
    _c("EEFFR", PCT, "other", desc="Efficiency ratio, YTD (published)"),
    _c("EEFFQR", PCT, "other", desc="Efficiency ratio, quarterly (published)"),
    _c("NONIIAY", PCT, "other", desc="Noninterest income / average assets (published)"),
    _c("LNLSDEPR", PCT, "other", desc="Net loans and leases / deposits (published)"),
    _c("LNLSNTV", PCT, "other", desc="Net loans and leases / assets (published)"),
)

# Spec-suggested codes that do not exist on the live endpoint. They are checked so the
# report says so explicitly; none of them is pulled.
LEGACY_CANDIDATES: tuple[Candidate, ...] = tuple(
    Candidate(code=c, unit=K, group="other", required=False)
    for c in (
        "SCHTM",
        "SCAFFV",
        "SCHTMFV",
        "RBCTOT",
        "TD250K",
        "ELECTCBLR",
        "DEPUNS",
        "FRBHLBADV",
        "IDNCLNLS",
    )
)

# The spec's "find code" items. ``pattern`` is matched (case-insensitively) against
# ``code + title + description``; ``chosen`` is the code selected after reading the hits.
KEYWORD_SEARCHES: tuple[KeywordSearch, ...] = (
    KeywordSearch(
        label="Tier 1 capital, dollars",
        pattern=r"TIER 1 RBC|TIER 1 CAPITAL",
        chosen="RBCT1J",
        unit=K,
        group="capital",
        description="Tier 1 capital, dollars (PCA definition, allowance-adjusted)",
    ),
    KeywordSearch(
        label="Total risk-based capital, dollars",
        pattern=r"RBC-TOTAL|TOTAL RISK.BASED CAPITAL",
        chosen="RBC",
        unit=K,
        group="capital",
        description="Total risk-based capital, dollars (PCA definition)",
    ),
    KeywordSearch(
        label="Held-to-maturity securities at amortized cost",
        pattern=r"HELD.TO.MATURITY|SECURITIES-H",
        chosen="SCHA",
        unit=K,
        group="securities",
        prototype="P2",
        description="Held-to-maturity securities at amortized cost",
    ),
)


def parse_dictionary(path: Path = DEFAULT_DICTIONARY_PATH) -> dict[str, dict[str, Any]]:
    """Read an FDIC ``*_properties.yaml`` file and return ``properties.data.properties``.

    The mapping is keyed by code; each value has ``title``, ``description`` and ``type``.
    """
    with Path(path).open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return doc["properties"]["data"]["properties"]


def entry_text(code: str, entry: dict[str, Any]) -> str:
    """The searchable text of one dictionary entry: code, title and description."""
    parts = [code, str(entry.get("title") or ""), str(entry.get("description") or "")]
    return " ".join(parts).replace("\n", " ")


def search_dictionary(
    dictionary: dict[str, dict[str, Any]], pattern: str, limit: int = 12
) -> list[tuple[str, str]]:
    """Return ``(code, title)`` pairs whose code/title/description match ``pattern``."""
    rx = re.compile(pattern, re.IGNORECASE)
    hits = [
        (code, str(entry.get("title") or ""))
        for code, entry in dictionary.items()
        if rx.search(entry_text(code, entry))
    ]
    return hits[:limit]


def _title(entry: dict[str, Any] | None) -> str:
    return str((entry or {}).get("title") or "").strip()


def build_field_map(dictionary: dict[str, dict[str, Any]]) -> FieldMapResult:
    """Resolve the candidate catalog and keyword searches against a dictionary.

    Codes missing from the dictionary are listed in ``not_found``; a missing *required*
    candidate is still kept out of the map (the API would silently drop it anyway), except
    for codes flagged in ``LIVE_ONLY``, which the live endpoint returns although the
    published dictionary omits them.
    """
    fields: list[FieldSpec] = []
    seen: set[str] = set()
    result = FieldMapResult(fields=fields)

    for cand in (*CANDIDATES, *LEGACY_CANDIDATES):
        entry = dictionary.get(cand.code)
        in_dict = entry is not None
        if not in_dict and cand.code not in LIVE_ONLY:
            result.not_found.append(cand.code)
            continue
        if not cand.required:
            continue
        if not in_dict:
            result.notes.append(
                f"{cand.code}: absent from risview_properties.yaml but returned by the live "
                "/financials endpoint; kept with in_dictionary=false."
            )
        fields.append(
            FieldSpec(
                code=cand.code,
                column=cand.code.lower(),
                description=cand.description or _title(entry) or cand.code,
                unit=cand.unit,
                group=cand.group,
                prototype=cand.prototype,
                in_dictionary=in_dict,
            )
        )
        seen.add(cand.code)

    for search in KEYWORD_SEARCHES:
        hits = search_dictionary(dictionary, search.pattern)
        result.search_report[search.label] = hits
        entry = dictionary.get(search.chosen)
        if entry is None:
            result.not_found.append(search.chosen)
            result.notes.append(
                f"{search.label}: chosen code {search.chosen} is not in the dictionary."
            )
            continue
        if search.chosen in seen:
            continue
        fields.append(
            FieldSpec(
                code=search.chosen,
                column=search.chosen.lower(),
                description=search.description or _title(entry),
                unit=search.unit,
                group=search.group,
                prototype=search.prototype,
            )
        )
        seen.add(search.chosen)
        result.notes.append(
            f"{search.label}: chose {search.chosen} ({_title(entry)}) from "
            f"{len(hits)} keyword hit(s)."
        )
    return result


# Verified on the live /financials endpoint on 2026-09-25 although the dictionary omits it.
LIVE_ONLY: frozenset[str] = frozenset({"FED_RSSD"})


def to_yaml_document(
    result: FieldMapResult, source: str = "risview_properties.yaml"
) -> dict[str, Any]:
    """Serialisable form of a field map, with stable key order for deterministic output."""
    return {
        "source": source,
        "fields": [
            {
                "code": f.code,
                "column": f.column,
                "description": f.description,
                "unit": f.unit,
                "group": f.group,
                "prototype": f.prototype,
                "first_available": f.first_available,
                "in_dictionary": f.in_dictionary,
            }
            for f in result.fields
        ],
        "not_found": sorted(result.not_found),
        "notes": list(result.notes),
    }


def write_field_map(result: FieldMapResult, path: Path = DEFAULT_FIELD_MAP_PATH) -> None:
    """Write ``config/fields.yaml`` (header comment + YAML); byte-identical on re-runs."""
    header = (
        "# Verified FDIC /financials field map. Generated by scripts/build_field_map.py;\n"
        "# edit the candidate catalog in src/bankcanary/fields.py, not this file.\n"
        "# first_available is filled by the data-quality step.\n"
    )
    body = yaml.safe_dump(to_yaml_document(result), sort_keys=False, allow_unicode=True, width=100)
    Path(path).write_text(header + body, encoding="utf-8")


def load_field_map(path: Path = DEFAULT_FIELD_MAP_PATH) -> list[FieldSpec]:
    """Load ``config/fields.yaml`` as :class:`FieldSpec` objects, validating units/groups."""
    with Path(path).open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    specs: list[FieldSpec] = []
    for row in doc["fields"]:
        if row["unit"] not in UNITS:
            raise ValueError(f"{row['code']}: unknown unit {row['unit']!r}")
        if row["group"] not in GROUPS:
            raise ValueError(f"{row['code']}: unknown group {row['group']!r}")
        specs.append(FieldSpec(**{k: row[k] for k in FieldSpec.__dataclass_fields__ if k in row}))
    codes = [s.code for s in specs]
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate codes in field map")
    return specs


def financial_codes(specs: list[FieldSpec] | None = None) -> list[str]:
    """Codes carrying financial values (everything outside identity/attribute groups)."""
    specs = load_field_map() if specs is None else specs
    return [s.code for s in specs if s.group not in ATTRIBUTE_GROUPS]


def attribute_codes(specs: list[FieldSpec] | None = None) -> list[str]:
    """Identity and attribute codes (cert, report date, name, state, class, RSSD ids...)."""
    specs = load_field_map() if specs is None else specs
    return [s.code for s in specs if s.group in ATTRIBUTE_GROUPS]


def columns_by_group(group: str, specs: list[FieldSpec] | None = None) -> list[str]:
    """Lower-case ``financials_raw`` column names for one group (e.g. ``income_ytd``)."""
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; expected one of {GROUPS}")
    specs = load_field_map() if specs is None else specs
    return [s.column for s in specs if s.group == group]
