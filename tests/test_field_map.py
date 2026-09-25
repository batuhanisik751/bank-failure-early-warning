"""Field-map resolution against a trimmed copy of the FDIC financials dictionary."""

from __future__ import annotations

from pathlib import Path

import pytest

from bankcanary import fields as fm

FIXTURE = Path(__file__).parent / "fixtures" / "fdic" / "risview_properties_sample.yaml"


@pytest.fixture(scope="module")
def dictionary() -> dict:
    return fm.parse_dictionary(FIXTURE)


@pytest.fixture(scope="module")
def result(dictionary: dict) -> fm.FieldMapResult:
    return fm.build_field_map(dictionary)


def test_parse_dictionary_returns_code_entries(dictionary: dict) -> None:
    assert len(dictionary) == 32
    assert dictionary["ASSET"]["title"] == "Total assets"


def test_found_candidates_are_mapped_with_lowercase_columns(result: fm.FieldMapResult) -> None:
    by_code = {f.code: f for f in result.fields}
    assert by_code["ASSET"].column == "asset"
    assert by_code["ASSET"].unit == "thousands_usd"
    assert by_code["ASSET"].group == "balance_sheet"
    assert by_code["RBC1AAJ"].unit == "percent"
    assert by_code["CERT"].group == "identity"
    assert by_code["NAME"].in_dictionary is True


def test_missing_codes_are_reported_and_excluded(result: fm.FieldMapResult) -> None:
    codes = {f.code for f in result.fields}
    assert "LNRECONS" not in codes  # a real candidate, not in the trimmed fixture
    assert "LNRECONS" in result.not_found
    assert "RBCTOT" in result.not_found  # legacy spec guess, never pulled
    assert "RBCTOT" not in codes


def test_live_only_code_is_kept_with_note(result: fm.FieldMapResult) -> None:
    by_code = {f.code: f for f in result.fields}
    assert by_code["FED_RSSD"].in_dictionary is False
    assert "FED_RSSD" not in result.not_found
    assert any(note.startswith("FED_RSSD") for note in result.notes)


def test_keyword_searches_resolve_to_chosen_codes(result: fm.FieldMapResult) -> None:
    by_code = {f.code: f for f in result.fields}
    assert by_code["RBCT1J"].unit == "thousands_usd"
    assert by_code["RBC"].group == "capital"
    assert by_code["SCHA"].prototype == "P2"
    hits = dict(result.search_report["Held-to-maturity securities at amortized cost"])
    assert {"SCHA", "SCHF", "SCHAR"} <= set(hits)
    assert sum(f.code == "RBCT1J" for f in result.fields) == 1  # no duplicate entries


def test_income_items_carry_income_ytd_group(result: fm.FieldMapResult) -> None:
    ytd = {f.code for f in result.fields if f.group == "income_ytd"}
    assert {"NETINC", "INTINC", "EINTEXP", "NTLNLS", "ELNATR"} <= ytd
    assert "ASSET" not in ytd


def test_write_and_load_roundtrip(result: fm.FieldMapResult, tmp_path: Path) -> None:
    out = tmp_path / "fields.yaml"
    fm.write_field_map(result, out)
    first = out.read_bytes()
    fm.write_field_map(result, out)
    assert out.read_bytes() == first  # deterministic
    specs = fm.load_field_map(out)
    assert [s.code for s in specs] == [f.code for f in result.fields]
    assert all(s.first_available is None for s in specs)
    assert "RBCTOT" in fm.to_yaml_document(result)["not_found"]


def test_helpers_split_attributes_from_financials(result: fm.FieldMapResult) -> None:
    specs = result.fields
    attrs = fm.attribute_codes(specs)
    fins = fm.financial_codes(specs)
    assert attrs[:2] == ["CERT", "REPDTE"]
    assert "NAME" in attrs and "NAME" not in fins
    assert "ASSET" in fins and "ASSET" not in attrs
    assert set(attrs).isdisjoint(fins)
    assert fm.columns_by_group("securities", specs) == ["schf", "scha"]
    with pytest.raises(ValueError):
        fm.columns_by_group("nope", specs)


def test_load_rejects_bad_unit(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "fields:\n- code: X\n  column: x\n  description: d\n  unit: furlongs\n"
        "  group: other\n  prototype: P1\n  first_available: null\n"
    )
    with pytest.raises(ValueError):
        fm.load_field_map(bad)
