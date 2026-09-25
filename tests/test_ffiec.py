"""FFIEC bulk download and schedule parser: MockTransport plus hand-written fixtures, no network."""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
import pytest

from bankcanary.sources import ffiec

FIXTURES = Path(__file__).parent / "fixtures" / "ffiec"

FORM_PAGE = """
<form method="post" action="./DownloadBulkData.aspx" id="form1">
<input type="hidden" name="__EVENTTARGET" id="__EVENTTARGET" value="" />
<input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="VS1" />
<select name="ctl00$MainContentHolder$DatesDropDownList" id="DatesDropDownList"></select>
</form>
"""
LISTED_PAGE = """
<input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="VS2" />
<select name="ctl00$MainContentHolder$DatesDropDownList" id="DatesDropDownList">
    <option selected="selected" value="152">06/30/2026</option>
    <option value="104">12/31/2022</option>
</select>
"""


@pytest.fixture
def bulk_zip(tmp_path: Path) -> Path:
    path = tmp_path / "call_single_period_2022-12-31.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for member in sorted(FIXTURES.glob("*.txt")):
            zf.write(member, arcname=member.name)
    return path


def test_hidden_fields_and_period_options():
    assert ffiec.hidden_fields(FORM_PAGE) == {"__EVENTTARGET": "", "__VIEWSTATE": "VS1"}
    assert ffiec.period_options(FORM_PAGE) == {}
    assert ffiec.period_options(LISTED_PAGE) == {"06/30/2026": "152", "12/31/2022": "104"}
    assert ffiec.period_label("2022-12-31") == "12/31/2022"


def test_download_single_period_round_trip(tmp_path: Path, bulk_zip: Path):
    posts: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=FORM_PAGE)
        form = dict(httpx.QueryParams(request.content.decode()))
        posts.append(form)
        if form["__EVENTTARGET"] == ffiec.FIELD_PRODUCT:
            assert form["__VIEWSTATE"] == "VS1"
            return httpx.Response(200, text=LISTED_PAGE)
        assert form["__VIEWSTATE"] == "VS2"
        assert form[ffiec.FIELD_PERIOD] == "104"
        assert form[ffiec.FIELD_FORMAT] == "TSVRadioButton"
        return httpx.Response(
            200,
            content=bulk_zip.read_bytes(),
            headers={"content-disposition": 'attachment; filename="Call Bulk 12312022.zip"'},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "raw"
    out = ffiec.download_single_period("2022-12-31", dest, client=client)
    assert out == dest / "call_single_period_2022-12-31.zip"
    assert out.read_bytes() == bulk_zip.read_bytes()
    assert len(posts) == 2
    # Cached: no further requests.
    assert ffiec.download_single_period("2022-12-31", dest, client=client) == out
    assert len(posts) == 2


def test_download_rejects_unknown_period_and_non_zip(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=FORM_PAGE)
        form = dict(httpx.QueryParams(request.content.decode()))
        if form["__EVENTTARGET"] == ffiec.FIELD_PRODUCT:
            return httpx.Response(200, text=LISTED_PAGE)
        return httpx.Response(200, text="<html>error</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="does not list period 03/31/2001"):
        ffiec.download_single_period("2001-03-31", tmp_path, client=client)
    with pytest.raises(RuntimeError, match="no ZIP"):
        ffiec.download_single_period("2022-12-31", tmp_path, client=client)
    assert not (tmp_path / "call_single_period_2022-12-31.zip").exists()


def test_parse_schedule_joins_parts_and_types_columns(bulk_zip: Path):
    rcb = ffiec.parse_schedule(bulk_zip, "RC-B")
    assert rcb.index.name == "IDRSSD" and rcb.index.tolist() == [37, 802866, 2942690]
    assert "IDRSSD" not in rcb.columns and not any(c.startswith("Unnamed") for c in rcb.columns)
    # Part 2 columns are joined on IDRSSD.
    assert {"RCFD1754", "RCON1754", "RCFD8500", "TEXT9999"} <= set(rcb.columns)
    assert rcb.at[802866, "RCFD1754"] == 91_327_000
    assert rcb.at[802866, "RCFD1773"] == 25_976_000 and rcb.at[802866, "RCON1773"] == 21_975_000
    assert rcb["RCFD1771"].isna().tolist() == [True, False, True]
    assert rcb.at[2942690, "RCON8500"] == 2000 and rcb["RCFD8500"].dtype == "float64"
    # A column with free text stays text; CONF and blanks are missing.
    assert rcb.at[802866, "TEXT9999"] == "some note"
    assert rcb["TEXT9999"].isna().tolist() == [True, False, True]
    assert rcb.attrs["descriptions"]["RCFD1754"] == "HELD-TO-MATURITY SECURITIES"


def test_parse_schedule_rco_and_confidential_cells(bulk_zip: Path):
    rco = ffiec.parse_schedule(bulk_zip, "RCO")
    assert rco.at[802866, "RCON5597"] == 151_592_000
    assert rco.at[37, "RCONF236"] == 123 and rco["RCONF236"].isna().sum() == 2
    with pytest.raises(KeyError, match="RC-Z"):
        ffiec.parse_schedule(bulk_zip, "RC-Z")


def test_schedule_members_orders_parts(bulk_zip: Path):
    with zipfile.ZipFile(bulk_zip) as zf:
        names = ffiec.schedule_members(zf, "rc-b")
    assert [n[-12:] for n in names] == ["(1 of 2).txt", "(2 of 2).txt"]
    with zipfile.ZipFile(bulk_zip) as zf:
        assert ffiec.schedule_members(zf, "RC") == []


def test_parse_schedule_ignores_trailing_tab_column(tmp_path: Path):
    # The published files end every row with a tab, which pandas reads as an unnamed column.
    text = '"IDRSSD"\tRCON5597\t\n\tESTIMATE OF UNINSURED DEPOSITS\t\n802866\t151592000\t\n37\t\t\n'
    path = tmp_path / "call_single_period_2022-12-31.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("FFIEC CDR Call Schedule RCO 12312022.txt", text)
    rco = ffiec.parse_schedule(path, "RC-O")
    assert rco.columns.tolist() == ["RCON5597"]
    assert rco["RCON5597"].tolist()[1] == 151_592_000 and rco["RCON5597"].isna().tolist() == [
        True,
        False,
    ]
