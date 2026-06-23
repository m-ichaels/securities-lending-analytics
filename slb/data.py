"""Loaders and the DuckDB store: universe, prices, shares outstanding, the funding rate, FINRA short interest, the Reg SHO
threshold lists, SEC fails to deliver, IB shortable snapshots when present, and the point-in-time symbol map built from
the CUSIP-symbol pairs in the fails data."""
from __future__ import annotations

import csv
import datetime as dt
import glob
import io
import os
import zipfile

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw"); DER = os.path.join(ROOT, "data", "derived"); REF = os.path.join(ROOT, "data", "reference")
DB_PATH = os.path.join(DER, "slb.duckdb")


def sql_path(p: str) -> str:
    return p.replace(os.sep, "/").replace("'", "''")


def load_universe() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, "data", "universe.csv"))


def load_prices(symbols: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    con = duckdb.connect(); q = f"select * from read_parquet('{sql_path(os.path.join(DER, 'prices.parquet'))}')"; conds = []
    if symbols:
        conds.append("symbol in (" + ",".join(f"'{s}'" for s in symbols) + ")")
    if start:
        conds.append(f"date >= '{start}'")
    if conds:
        q += " where " + " and ".join(conds)
    df = con.execute(q + " order by symbol, date").df(); con.close(); return df


def load_events() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DER, "events_yahoo.csv"))


def load_shares() -> pd.DataFrame:
    p = os.path.join(REF, "shares_outstanding.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["symbol", "market_cap", "price", "shares_outstanding", "sector", "industry", "as_of"])


def load_effr() -> pd.Series:
    t = pd.read_csv(os.path.join(REF, "effr.csv")); s = pd.Series(t["effr"].values / 100.0, index=pd.to_datetime(t["date"])); return s.sort_index()


# ---- FINRA short interest ---------------------------------------------------------------------------------------------
def load_short_interest(symbols: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "finra", "shrt*.csv"))):
        d = os.path.basename(p)[4:12]
        with open(p, encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f, delimiter="|"):
                s = r["symbolCode"]
                if symbols is not None and s not in symbols:
                    continue
                try:
                    rows.append((r["settlementDate"] or f"{d[:4]}-{d[4:6]}-{d[6:]}", s, float(r["currentShortPositionQuantity"] or 0), float(r["previousShortPositionQuantity"] or 0), float(r["averageDailyVolumeQuantity"] or 0), float(r["daysToCoverQuantity"] or 0), r["marketClassCode"]))
                except ValueError:
                    continue
    df = pd.DataFrame(rows, columns=["settle_date", "symbol", "shares_short", "prev_short", "adv", "days_to_cover", "market"])
    df["settle_date"] = pd.to_datetime(df["settle_date"]); return df.sort_values(["symbol", "settle_date"]).drop_duplicates(["symbol", "settle_date"])


# ---- Regulation SHO threshold lists ----------------------------------------------------------------------------------
def load_threshold(symbols: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "threshold", "nasdaqth*.txt"))) + sorted(glob.glob(os.path.join(RAW, "threshold", "nyseth*.csv"))):
        name = os.path.basename(p); d = name[8:16] if name.startswith("nasdaq") else name[6:14]
        try:
            date = dt.date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        except ValueError:
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 4 or parts[0] in ("Symbol", "") or parts[0][:1].isdigit():
                    continue
                if symbols is None or parts[0] in symbols:
                    rows.append((date, parts[0], parts[2], "nasdaq" if name.startswith("nasdaq") else "nyse"))
    df = pd.DataFrame(rows, columns=["date", "symbol", "market", "source"]); df["date"] = pd.to_datetime(df["date"]); return df.drop_duplicates(["date", "symbol"])


MIN_THRESHOLD_COVERAGE = 0.8


def threshold_with_coverage_rule(th: pd.DataFrame) -> pd.DataFrame:
    """a source whose daily lists cover less than 80 % of the days the other source covers is left out, so that a partial
    download (the NYSE endpoint allows about 80 requests an hour) does not flag one exchange's names for four months only"""
    avail = threshold_days_available()
    for src, key in (("nasdaq", "nasdaq_days"), ("nyse", "nyse_days")):
        other = avail["nyse_days"] if src == "nasdaq" else avail["nasdaq_days"]
        if other and avail[key] < MIN_THRESHOLD_COVERAGE * other and len(th):
            th = th[th["source"] != src]
            print(f"threshold lists from {src} cover {avail[key]} days against {other}: left out of the model until the download is complete")
    return th


def threshold_days_available() -> dict:
    n = sorted(glob.glob(os.path.join(RAW, "threshold", "nasdaqth*.txt"))); y = sorted(glob.glob(os.path.join(RAW, "threshold", "nyseth*.csv")))
    return {"nasdaq_days": len(n), "nyse_days": len(y), "first": os.path.basename(n[0])[8:16] if n else None, "last": os.path.basename(n[-1])[8:16] if n else None}


# ---- SEC fails to deliver -------------------------------------------------------------------------------------------
def load_ftd(symbols: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "ftd", "cnsfails*.zip"))):
        try:
            z = zipfile.ZipFile(p)
        except zipfile.BadZipFile:
            continue
        for n in z.namelist():
            for line in io.TextIOWrapper(z.open(n), encoding="latin-1"):
                parts = line.rstrip("\r\n").split("|")
                if len(parts) < 6 or not parts[0].isdigit():
                    continue
                if symbols is None or parts[2] in symbols:
                    try:
                        rows.append((dt.date(int(parts[0][:4]), int(parts[0][4:6]), int(parts[0][6:8])), parts[1], parts[2], float(parts[3] or 0), float(parts[5] or 0) if parts[5] not in ("", ".") else np.nan))
                    except ValueError:
                        continue
    df = pd.DataFrame(rows, columns=["settle_date", "cusip", "symbol", "quantity", "price"]); df["settle_date"] = pd.to_datetime(df["settle_date"]); return df


def symbol_map_from_ftd(universe: set[str]) -> pd.DataFrame:
    """Point-in-time symbol changes: a CUSIP that appears under two symbols in the fails data links the old ticker to the new
    one (FB -> META, and every rename in the window).  Returns old_symbol, symbol, first date seen under the new symbol."""
    pairs = {}
    for p in sorted(glob.glob(os.path.join(RAW, "ftd", "cnsfails*.zip"))):
        try:
            z = zipfile.ZipFile(p)
        except zipfile.BadZipFile:
            continue
        for n in z.namelist():
            for line in io.TextIOWrapper(z.open(n), encoding="latin-1"):
                parts = line.rstrip("\r\n").split("|")
                if len(parts) < 6 or not parts[0].isdigit():
                    continue
                pairs.setdefault(parts[1], {}).setdefault(parts[2], parts[0])
    rows = []
    for cusip, syms in pairs.items():
        if len(syms) < 2:
            continue
        ordered = sorted(syms.items(), key=lambda kv: kv[1]); newest = ordered[-1][0]
        if newest in universe:
            for old, first in ordered[:-1]:
                if old != newest and not placeholder(old, newest):
                    rows.append((old, newest, cusip, f"{ordered[-1][1][:4]}-{ordered[-1][1][4:6]}-{ordered[-1][1][6:]}"))
    return pd.DataFrame(rows, columns=["old_symbol", "symbol", "cusip", "valid_from"]).drop_duplicates(["old_symbol", "symbol"])


def placeholder(old: str, new: str) -> bool:
    """the CNS file carries temporary tickers around corporate actions (XXXXZZZZ while a reverse split settles, XXXXD for
    the post-split security for twenty days); those are not renames"""
    return old.endswith("ZZZZ") or old == new + "D" or len(old) > 5


def relabel(df: pd.DataFrame, date_col: str, smap: pd.DataFrame) -> pd.DataFrame:
    """point-in-time: rows under an old ticker before the rename date are moved to the current ticker (a ticker can be
    reused by another company later, as CCC was by Clarivate and then CCC Intelligent Solutions, so the date matters)"""
    if not len(smap):
        return df
    df = df.copy(); df[date_col] = pd.to_datetime(df[date_col])
    for r in smap.itertuples():
        m = (df["symbol"] == r.old_symbol) & (df[date_col] < pd.Timestamp(r.valid_from))
        df.loc[m, "symbol"] = r.symbol
    return df


# ---- Interactive Brokers shortable snapshots (when the fetch workflow has run) ------------------------------------------
def load_ib(symbols: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "ib", "usa_*.txt"))):
        d = os.path.basename(p)[4:12]
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 8 or parts[0].startswith("#"):
                    continue
                sym = parts[0]
                if symbols is not None and sym not in symbols:
                    continue
                try:
                    rows.append((dt.date(int(d[:4]), int(d[4:6]), int(d[6:8])), sym, float(parts[5] or "nan"), float(parts[6] or "nan"), float(parts[7].replace(">", "") or 0)))
                except ValueError:
                    continue
    df = pd.DataFrame(rows, columns=["date", "symbol", "rebate_rate", "fee_rate", "available"]); df["date"] = pd.to_datetime(df["date"]); return df


# ---- store --------------------------------------------------------------------------------------------------------------
SCHEMA = """
create table if not exists securities (symbol varchar primary key, name varchar, market varchar, exchange varchar, shares_outstanding double, market_cap double, sector varchar, industry varchar);
create table if not exists prices (symbol varchar, date date, open double, high double, low double, close double, adjclose double, volume double);
create table if not exists short_interest (settle_date date, symbol varchar, shares_short double, prev_short double, adv double, days_to_cover double, market varchar);
create table if not exists threshold (date date, symbol varchar, market varchar, source varchar);
create table if not exists ftd (settle_date date, cusip varchar, symbol varchar, quantity double, price double);
create table if not exists symbol_history (old_symbol varchar, symbol varchar, cusip varchar, valid_from date);
create table if not exists ib_snapshots (date date, symbol varchar, rebate_rate double, fee_rate double, available double);
create table if not exists effr (date date, rate double);
create table if not exists borrow (date date, symbol varchar, sir double, days_to_cover double, on_threshold boolean, ftd_ratio double, score double, fee double, fee_floor_binds boolean, lendable double, utilisation double, p_locate double, recall_hazard double, source varchar);
create table if not exists accruals (date date, book varchar, symbol varchar, side varchar, notional double, rate double, day_count integer, amount double, kind varchar);
create table if not exists pb_statement (date date, book varchar, line_id varchar, symbol varchar, side varchar, notional double, rate double, day_count integer, amount double, kind varchar);
create table if not exists fin_breaks (date date, book varchar, break_type varchar, symbol varchar, key varchar, detail varchar, seeded varchar);
"""


def connect(path: str = DB_PATH, fresh: bool = False):
    if fresh and os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True); con = duckdb.connect(path)
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    return con


COMPACT = {"short_interest": os.path.join(DER, "short_interest.parquet"), "threshold": os.path.join(DER, "threshold.parquet"), "ftd": os.path.join(DER, "ftd.parquet"), "symbol_history": os.path.join(DER, "symbol_history.csv"), "ib": os.path.join(DER, "ib_snapshots.parquet")}


def raw_available() -> bool:
    return bool(glob.glob(os.path.join(RAW, "finra", "shrt*.csv")))


def compact_inputs(syms: set[str], from_raw: bool | None = None) -> dict:
    """the universe's slice of the raw files, relabelled point-in-time, written to data/derived so that the repository
    carries what the pipeline needs (the raw FINRA and SEC files are 550 MB) and read back from there when the raw files
    are absent"""
    from_raw = raw_available() if from_raw is None else from_raw
    if from_raw:
        smap = symbol_map_from_ftd(syms); olds = set(smap["old_symbol"]) if len(smap) else set()
        si = relabel(load_short_interest(syms | olds), "settle_date", smap); si = si[si["symbol"].isin(syms)].drop_duplicates(["symbol", "settle_date"])
        th = relabel(load_threshold(syms | olds), "date", smap); th = th[th["symbol"].isin(syms)].drop_duplicates(["date", "symbol"])
        th = threshold_with_coverage_rule(th)
        ft = relabel(load_ftd(syms | olds), "settle_date", smap); ft = ft[ft["symbol"].isin(syms)]
        ib = load_ib(syms)
        os.makedirs(DER, exist_ok=True)
        si.to_parquet(COMPACT["short_interest"], index=False); th.to_parquet(COMPACT["threshold"], index=False); ft.to_parquet(COMPACT["ftd"], index=False)
        smap.to_csv(COMPACT["symbol_history"], index=False, lineterminator="\n")
        if len(ib):
            ib.to_parquet(COMPACT["ib"], index=False)
    else:
        si = pd.read_parquet(COMPACT["short_interest"]); th = pd.read_parquet(COMPACT["threshold"]); ft = pd.read_parquet(COMPACT["ftd"])
        smap = pd.read_csv(COMPACT["symbol_history"]) if os.path.exists(COMPACT["symbol_history"]) else pd.DataFrame(columns=["old_symbol", "symbol", "cusip", "valid_from"])
        ib = pd.read_parquet(COMPACT["ib"]) if os.path.exists(COMPACT["ib"]) else pd.DataFrame(columns=["date", "symbol", "rebate_rate", "fee_rate", "available"])
    return {"short_interest": si, "threshold": th, "ftd": ft, "symbol_history": smap, "ib": ib, "from_raw": from_raw}


def build_store(con, verbose: bool = True, from_raw: bool | None = None) -> dict:
    uni = load_universe(); sh = load_shares(); u = uni.merge(sh[["symbol", "shares_outstanding", "market_cap", "sector", "industry"]], on="symbol", how="left")
    con.execute("delete from securities"); con.execute("insert into securities select symbol, name, market, exchange, shares_outstanding, market_cap, sector, industry from u")
    con.execute("delete from prices"); con.execute(f"insert into prices select symbol, cast(date as date), open, high, low, close, adjclose, volume from read_parquet('{sql_path(os.path.join(DER, 'prices.parquet'))}')")
    syms = set(uni["symbol"])
    c = compact_inputs(syms, from_raw); smap, si, th, ft, ib = c["symbol_history"], c["short_interest"], c["threshold"], c["ftd"], c["ib"]
    con.execute("delete from symbol_history")
    if len(smap):
        con.execute("insert into symbol_history select old_symbol, symbol, cusip, cast(valid_from as date) from smap")
    con.execute("delete from short_interest"); con.execute("insert into short_interest select cast(settle_date as date), symbol, shares_short, prev_short, adv, days_to_cover, market from si")
    con.execute("delete from threshold")
    if len(th):
        con.execute("insert into threshold select cast(date as date), symbol, market, source from th")
    con.execute("delete from ftd")
    if len(ft):
        con.execute("insert into ftd select cast(settle_date as date), cusip, symbol, quantity, price from ft")
    con.execute("delete from ib_snapshots")
    if len(ib):
        con.execute("insert into ib_snapshots select cast(date as date), symbol, rebate_rate, fee_rate, available from ib")
    e = load_effr(); ef = pd.DataFrame({"date": e.index.date, "rate": e.values}); con.execute("delete from effr"); con.execute("insert into effr select cast(date as date), rate from ef")
    out = store_counts(con); out["from_raw"] = c["from_raw"]
    if verbose:
        print(out)
    return out


def store_counts(con) -> dict:
    """what is in the store, for the results and the README"""
    q = lambda sql: con.execute(sql).fetchone()
    th = q("select count(*), count(distinct date) from threshold"); si = q("select count(*), count(distinct settle_date) from short_interest"); ib = q("select count(*), count(distinct date) from ib_snapshots")
    return {"securities": q("select count(*) from securities")[0], "with_shares": q("select count(*) from securities where shares_outstanding is not null")[0], "prices": q("select count(*) from prices")[0], "short_interest_rows": si[0], "short_interest_dates": si[1], "threshold_rows": th[0], "threshold_days": th[1],
            "ftd_rows": q("select count(*) from ftd")[0], "symbol_changes": q("select count(*) from symbol_history")[0], "ib_rows": ib[0], "ib_days": ib[1], "effr_days": q("select count(*) from effr")[0],
            "threshold_sources_in_model": [r[0] for r in con.execute("select distinct source from threshold order by 1").fetchall()], "threshold_files_available": threshold_days_available() if raw_available() else None}
