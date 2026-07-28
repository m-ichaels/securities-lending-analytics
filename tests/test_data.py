"""Loaders and the point-in-time symbol map on small synthetic raw files, and the settlement-date calendar."""
import datetime as dt
import io
import os
import sys
import zipfile

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))

from slb import data as D  # noqa: E402
import download as DL  # noqa: E402


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """a tiny raw tree: two FINRA files, two Nasdaq threshold lists, one NYSE list, one CNS zip with a rename, one IB file"""
    monkeypatch.setattr(D, "RAW", str(tmp_path))
    f = tmp_path / "finra"; f.mkdir()
    head = "accountingYearMonthNumber|symbolCode|issueName|issuerServicesGroupExchangeCode|marketClassCode|currentShortPositionQuantity|previousShortPositionQuantity|changePreviousNumber|changePercent|averageDailyVolumeQuantity|daysToCoverQuantity|revisionFlag|settlementDate\n"
    (f / "shrt20240115.csv").write_text(head + "202401|AAA|Alpha Inc|N|NYSE|1000000|900000|100000|11.1|250000|4.0||2024-01-15\n202401|FB|Old Meta|Q|NNM|5000000|4000000|1000000|25|1000000|5.0||2024-01-15\n")
    (f / "shrt20240131.csv").write_text(head + "202401|AAA|Alpha Inc|N|NYSE|1200000|1000000|200000|20|300000|4.0||2024-01-31\n202401|META|Meta|Q|NNM|6000000|5000000|1000000|20|1000000|6.0||2024-01-31\n")
    t = tmp_path / "threshold"; t.mkdir()
    (t / "nasdaqth20240116.txt").write_text("Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Rule 4320\nFB|Old Meta|Q|Y|N\n20240116|\n")
    (t / "nasdaqth20240117.txt").write_text("Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Rule 4320\nZZZ|Other|Q|Y|N\n")
    (t / "nyseth20240116.csv").write_text("Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Filler|Filler\nAAA|Alpha Inc|NYSE|Y||\n")
    z = tmp_path / "ftd"; z.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("cnsfails202401a.txt", "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20240110|30303M102|FB|1000|META PLATFORMS|350.10\n20240120|30303M102|META|2000|META PLATFORMS|360.00\n20240110|000000001|AAA|500|ALPHA INC|10.00\n20240110|000000009|AAAZZZZ|5|ALPHA PLACEHOLDER|.\n20240120|000000009|AAA|5|ALPHA|10.00\n")
    (z / "cnsfails202401a.zip").write_bytes(buf.getvalue())
    ib = tmp_path / "ib"; ib.mkdir()
    (ib / "usa_20240116.txt").write_text("#BOF|2024.01.16|08:00:00\n#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE\nAAA|USD|ALPHA INC|1|US000|4.5|0.75|>10000000\nMETA|USD|META|2|US001|-20.0|25.0|5000\n#EOF\n")
    return tmp_path


def test_short_interest_loader(raw):
    si = D.load_short_interest({"AAA", "FB", "META"})
    assert len(si) == 4 and set(si["symbol"]) == {"AAA", "FB", "META"}
    a = si[si["symbol"] == "AAA"].sort_values("settle_date")
    assert list(a["shares_short"]) == [1000000.0, 1200000.0] and a["settle_date"].iloc[0] == pd.Timestamp("2024-01-15")


def test_threshold_loader_skips_trailer_and_reads_both_exchanges(raw):
    th = D.load_threshold(None)
    assert set(zip(th["symbol"], th["source"])) == {("FB", "nasdaq"), ("ZZZ", "nasdaq"), ("AAA", "nyse")}
    assert th[th["symbol"] == "AAA"]["date"].iloc[0] == pd.Timestamp("2024-01-16")


def test_ftd_loader(raw):
    ft = D.load_ftd({"FB", "META", "AAA"})
    assert len(ft) == 4 and ft["quantity"].sum() == 3505
    assert ft[ft["symbol"] == "AAA"]["price"].iloc[0] == 10.0


def test_symbol_map_finds_rename_and_ignores_placeholders(raw):
    m = D.symbol_map_from_ftd({"META", "AAA"})
    assert list(m["old_symbol"]) == ["FB"] and list(m["symbol"]) == ["META"] and m["valid_from"].iloc[0] == "2024-01-20"
    assert D.placeholder("AAAZZZZ", "AAA") and D.placeholder("BMNRD", "BMNR") and not D.placeholder("FB", "META")


def test_relabel_is_point_in_time():
    smap = pd.DataFrame([{"old_symbol": "CCC", "symbol": "CLVT", "cusip": "x", "valid_from": "2021-02-02"}])
    df = pd.DataFrame({"symbol": ["CCC", "CCC", "CLVT"], "settle_date": ["2020-06-15", "2025-12-15", "2022-01-14"], "v": [1, 2, 3]})
    out = D.relabel(df, "settle_date", smap)
    assert list(out["symbol"]) == ["CLVT", "CCC", "CLVT"]      # the 2025 CCC is a different company and keeps its ticker


def test_ib_loader(raw):
    ib = D.load_ib(None)
    assert len(ib) == 2
    meta = ib[ib["symbol"] == "META"].iloc[0]
    assert meta["fee_rate"] == 25.0 and meta["rebate_rate"] == -20.0 and meta["available"] == 5000
    assert ib[ib["symbol"] == "AAA"]["available"].iloc[0] == 10000000


def test_compact_round_trip(raw, monkeypatch, tmp_path):
    der = tmp_path / "derived"; der.mkdir(); monkeypatch.setattr(D, "DER", str(der))
    monkeypatch.setattr(D, "COMPACT", {k: os.path.join(str(der), os.path.basename(v)) for k, v in D.COMPACT.items()})
    a = D.compact_inputs({"AAA", "META"}, from_raw=True); b = D.compact_inputs({"AAA", "META"}, from_raw=False)
    assert a["from_raw"] and not b["from_raw"]
    # FB rows before the rename are relabelled to META in both
    assert set(a["short_interest"]["symbol"]) == {"AAA", "META"} and len(a["short_interest"]) == 4
    # the NYSE list covers one day of the two Nasdaq days (50 % < 80 %), so it is left out until the download is complete
    assert len(b["short_interest"]) == 4 and len(b["threshold"]) == len(a["threshold"]) == 1 and set(a["threshold"]["source"]) == {"nasdaq"} and len(b["ftd"]) == len(a["ftd"])
    assert list(b["symbol_history"]["old_symbol"]) == ["FB"]


def test_threshold_coverage_rule(raw):
    th = D.load_threshold({"AAA", "FB", "META", "ZZZ"})
    assert set(th["source"]) == {"nasdaq", "nyse"}
    assert set(D.threshold_with_coverage_rule(th)["source"]) == {"nasdaq"}
    (raw / "threshold" / "nyseth20240117.csv").write_text("Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Filler|Filler\nAAA|Alpha Inc|NYSE|Y||\n")
    th2 = D.load_threshold({"AAA", "FB", "META", "ZZZ"})
    assert set(D.threshold_with_coverage_rule(th2)["source"]) == {"nasdaq", "nyse"}       # two of two days: kept


def test_settlement_dates_rule():
    d = DL.settlement_dates(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    # 15 Jan 2024 is a Monday (MLK day is a holiday), so the settlement moves to Friday 12 Jan; 31 Jan is a Wednesday
    assert d[0] == dt.date(2024, 1, 12) and d[1] == dt.date(2024, 1, 31)
    assert d[2] == dt.date(2024, 2, 15) and d[3] == dt.date(2024, 2, 29) and d[4] == dt.date(2024, 3, 15) and d[5] == dt.date(2024, 3, 28)   # 31 Mar is a Sunday and 29 Mar 2024 is Good Friday
    assert all(DL.is_bd(x) for x in d)


def test_nyse_holidays():
    assert not DL.is_bd(dt.date(2024, 7, 4)) and not DL.is_bd(dt.date(2025, 1, 9)) and DL.is_bd(dt.date(2024, 7, 5))
    assert not DL.is_bd(dt.date(2023, 6, 19)) and DL.is_bd(dt.date(2021, 6, 18))    # Juneteenth observed from 2022
