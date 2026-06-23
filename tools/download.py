#!/usr/bin/env python3
"""Free data for the securities-lending stack.   python tools/download.py [finra] [threshold] [threshold-nyse] [ftd] [prices] [shares] [rates] [all] [--from 2020-01-01]

  FINRA      bi-monthly equity short interest, every US listed security (settlement dates on the 15th and the last day of
             the month, moved back to the preceding business day)      -> data/raw/finra/shrt<date>.csv
  Nasdaq     daily Regulation SHO threshold list for Nasdaq-listed names  -> data/raw/threshold/nasdaqth<date>.txt
  NYSE       daily Regulation SHO threshold list for NYSE-listed names    -> data/raw/threshold/nyseth<date>.csv (when the API answers)
  SEC        half-monthly fails-to-deliver files (CNS)                    -> data/raw/ftd/cnsfails<yyyymm>[ab].zip
  Yahoo      daily bars, dividends and splits for the universe            -> data/raw/yahoo/<symbol>.json, data/derived/prices.parquet
  IB         the shortable-stock file (fee, rebate, available) arrives through .github/workflows/fetch-ib.yml because the
             FTP port is blocked on most office networks; tools/build.py reads whatever snapshots are in data/raw/ib/
The universe (data/universe.csv) is the exchange-listed common stocks with the largest short positions and the largest
average daily volumes in the latest FINRA file, so it contains both the general-collateral names and the hard-to-borrow ones.
"""
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw"); DER = os.path.join(ROOT, "data", "derived")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
SEC_UA = "ProjectG securities-lending research contact@example.com"
MONTH_CODES = "FGHJKMNQUVXZ"


def get(url, retries=3, timeout=120, ua=UA, wait_429=True):
    k = 0
    while k < retries:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and wait_429:
                # Cloudflare rate limit: honour Retry-After (the NYSE API asks for about an hour) and try again
                wait = min(int(e.headers.get("Retry-After", "600") or 600), 3600) + 5
                print(f"  429 from {url[:60]}: waiting {wait}s", flush=True); time.sleep(wait); continue
            if e.code in (403, 404, 500):
                return None
            if k == retries - 1:
                return None
            time.sleep(2 * (k + 1))
        except Exception:  # noqa: BLE001
            if k == retries - 1:
                return None
            time.sleep(2 * (k + 1))
        k += 1


# ---- calendars (weekday business days with the NYSE holidays; enough to find FINRA's settlement dates) ----------------
def easter(y):
    a = y % 19; b, c = divmod(y, 100); d, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3; h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
    mo, da = divmod(h + l - 7 * m + 114, 31); return dt.date(y, mo, da + 1)


def nth(y, m, wd, n):
    if n > 0:
        d = dt.date(y, m, 1); d += dt.timedelta(days=(wd - d.weekday()) % 7); return d + dt.timedelta(days=7 * (n - 1))
    d = dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1); return d - dt.timedelta(days=(d.weekday() - wd) % 7)


def obs(d):
    return d - dt.timedelta(days=1) if d.weekday() == 5 else d + dt.timedelta(days=1) if d.weekday() == 6 else d


def holidays(y):
    h = {obs(dt.date(y, 1, 1)) if dt.date(y, 1, 1).weekday() != 5 else None, nth(y, 1, 0, 3), nth(y, 2, 0, 3), easter(y) - dt.timedelta(days=2), nth(y, 5, 0, -1), obs(dt.date(y, 7, 4)), nth(y, 9, 0, 1), nth(y, 11, 3, 4), obs(dt.date(y, 12, 25))}
    if y >= 2022:
        h.add(obs(dt.date(y, 6, 19)))
    for md in {2018: ["12-05"], 2025: ["01-09"]}.get(y, []):
        h.add(dt.date.fromisoformat(f"{y}-{md}"))
    return {x for x in h if x}


def is_bd(d):
    return d.weekday() < 5 and d not in holidays(d.year)


def prev_bd(d):
    while not is_bd(d):
        d -= dt.timedelta(days=1)
    return d


def settlement_dates(start, end):
    out = []; y, m = start.year, start.month
    while dt.date(y, m, 1) <= end:
        last = dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)
        for d in (dt.date(y, m, 15), last):
            d = prev_bd(d)
            if start <= d <= end:
                out.append(d)
        m += 1
        if m == 13:
            m = 1; y += 1
    return out


# ---- FINRA short interest ---------------------------------------------------------------------------------------------
def finra(start, end):
    os.makedirs(os.path.join(RAW, "finra"), exist_ok=True); n = 0
    for d in settlement_dates(start, end):
        p = os.path.join(RAW, "finra", f"shrt{d.strftime('%Y%m%d')}.csv")
        if os.path.exists(p):
            n += 1; continue
        raw = None
        for back in range(0, 4):                       # FINRA occasionally settles a day earlier than the rule
            dd = d - dt.timedelta(days=back)
            raw = get(f"https://cdn.finra.org/equity/otcmarket/biweekly/shrt{dd.strftime('%Y%m%d')}.csv", retries=1)
            if raw:
                break
        if raw:
            open(p, "wb").write(raw); n += 1; print(f"  finra {d} ({len(raw) // 1024} KB)")
        else:
            print(f"  finra {d} missing")
        time.sleep(0.3)
    print(f"finra: {n} files")


# ---- Regulation SHO threshold lists ----------------------------------------------------------------------------------
NYSE_DELAY_S = 5.0     # the NYSE download endpoint answers 429 after about 80 requests in quick succession


def threshold(start, end, nasdaq=True, nyse=True):
    os.makedirs(os.path.join(RAW, "threshold"), exist_ok=True); n = 0; d = start
    while nasdaq and d <= end:
        if is_bd(d):
            p = os.path.join(RAW, "threshold", f"nasdaqth{d.strftime('%Y%m%d')}.txt")
            if not os.path.exists(p):
                raw = get(f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{d.strftime('%Y%m%d')}.txt", retries=1, timeout=60)
                if raw:
                    open(p, "wb").write(raw); n += 1
                time.sleep(0.15)
            else:
                n += 1
        d += dt.timedelta(days=1)
    if nasdaq:
        print(f"threshold (nasdaq): {n} daily files", flush=True)
    # NYSE: the regulatory download sits behind Cloudflare and rate-limits after about 80 quick requests (429 with a one-hour
    # Retry-After), so the loop is slow and waits out a 429 when it gets one; an empty list on a day is a real answer and is kept
    m = 0; d = start
    while nyse and d <= end:
        if is_bd(d):
            p = os.path.join(RAW, "threshold", f"nyseth{d.strftime('%Y%m%d')}.csv")
            if not os.path.exists(p):
                raw = get(f"https://www.nyse.com/api/regulatory/threshold-securities/download?selectedDate={d.strftime('%Y-%m-%d')}&market=", retries=2, timeout=60)
                if raw is not None and b"Symbol" in raw[:200]:
                    open(p, "wb").write(raw); m += 1
                    if m % 50 == 0:
                        print(f"  nyse {m} files, at {d}", flush=True)
                time.sleep(NYSE_DELAY_S)
            else:
                m += 1
        d += dt.timedelta(days=1)
    if nyse:
        print(f"threshold (nyse): {m} daily files", flush=True)


# ---- SEC fails to deliver ---------------------------------------------------------------------------------------------
def ftd(start, end):
    os.makedirs(os.path.join(RAW, "ftd"), exist_ok=True); n = 0; y, m = start.year, start.month
    while dt.date(y, m, 1) <= end:
        for half in "ab":
            p = os.path.join(RAW, "ftd", f"cnsfails{y}{m:02d}{half}.zip")
            if os.path.exists(p):
                n += 1; continue
            raw = get(f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{y}{m:02d}{half}.zip", retries=2, ua=SEC_UA)
            if raw:
                open(p, "wb").write(raw); n += 1
            time.sleep(0.4)
        m += 1
        if m == 13:
            m = 1; y += 1
    print(f"ftd: {n} half-monthly files")


# ---- universe from the latest FINRA file -------------------------------------------------------------------------------
def universe(n_short=600, n_volume=600):
    files = sorted(f for f in os.listdir(os.path.join(RAW, "finra")) if f.startswith("shrt"))
    if not files:
        raise SystemExit("download the FINRA files first")
    rows = list(csv.DictReader(open(os.path.join(RAW, "finra", files[-1]), encoding="utf-8", errors="replace"), delimiter="|"))
    ok = [r for r in rows if r["marketClassCode"] in ("NYSE", "NNM", "SC", "AMEX") and r["symbolCode"].isalpha() and len(r["symbolCode"]) <= 5
          and not any(w in r["issueName"].upper() for w in (" ETF", "TRUST", " FUND", "ISHARES", "SPDR", "PROSHARES", "DIREXION", "VANGUARD", "INVESCO", "ETN", "PREFERRED", " PFD", "WARRANT", " WT", "UNIT", "DEPOSITARY", " ADR", "NOTES", "DEBENTURE"))]
    for r in ok:
        r["si"] = float(r["currentShortPositionQuantity"] or 0); r["adv"] = float(r["averageDailyVolumeQuantity"] or 0)
    by_si = sorted(ok, key=lambda r: -r["si"])[:n_short]; by_adv = sorted(ok, key=lambda r: -r["adv"])[:n_volume]
    seen = {}
    for r in by_si + by_adv:
        seen.setdefault(r["symbolCode"], {"symbol": r["symbolCode"], "name": r["issueName"], "market": r["marketClassCode"], "exchange": "XNYS" if r["marketClassCode"] in ("NYSE", "AMEX") else "XNAS"})
    uni = sorted(seen.values(), key=lambda r: r["symbol"])
    with open(os.path.join(ROOT, "data", "universe.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "name", "market", "exchange"]); w.writeheader(); w.writerows(uni)
    print(f"universe: {len(uni)} names from {files[-1]}")
    return [u["symbol"] for u in uni]


# ---- Yahoo prices ---------------------------------------------------------------------------------------------------
def yahoo(sym, rng="10y"):
    raw = get(f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d&events=div,splits", retries=2, timeout=30)
    if raw is None:
        return None
    try:
        return json.loads(raw)["chart"]["result"][0]
    except Exception:  # noqa: BLE001
        return None


def prices(symbols):
    os.makedirs(os.path.join(RAW, "yahoo"), exist_ok=True); os.makedirs(DER, exist_ok=True)
    bars, events, missing = [], [], []
    for i, sym in enumerate(symbols):
        p = os.path.join(RAW, "yahoo", f"{sym}.json")
        if os.path.exists(p):
            r = json.load(open(p))
        else:
            r = yahoo(sym.replace(".", "-"))
            if r is None:
                missing.append(sym); continue
            json.dump(r, open(p, "w")); time.sleep(0.2)
        ts = r.get("timestamp") or []; q = r["indicators"]["quote"][0]; adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(ts))
        for k, t in enumerate(ts):
            c = q["close"][k]
            if c is None:
                continue
            d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
            bars.append((sym, d, q["open"][k] if q["open"][k] is not None else c, q["high"][k] if q["high"][k] is not None else c, q["low"][k] if q["low"][k] is not None else c, c, adj[k] if adj[k] is not None else c, q["volume"][k] or 0))
        ev = r.get("events", {})
        for t, e in (ev.get("dividends") or {}).items():
            events.append((sym, dt.datetime.fromtimestamp(int(e.get("date", t)), dt.timezone.utc).date().isoformat(), "dividend", e["amount"]))
        for t, e in (ev.get("splits") or {}).items():
            events.append((sym, dt.datetime.fromtimestamp(int(e.get("date", t)), dt.timezone.utc).date().isoformat(), "split", e["numerator"] / e["denominator"]))
        if (i + 1) % 100 == 0:
            print(f"  prices {i + 1}/{len(symbols)}", flush=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = list(zip(*bars))
    pq.write_table(pa.table({"symbol": cols[0], "date": cols[1], "open": pa.array(cols[2], pa.float64()), "high": pa.array(cols[3], pa.float64()), "low": pa.array(cols[4], pa.float64()), "close": pa.array(cols[5], pa.float64()), "adjclose": pa.array(cols[6], pa.float64()), "volume": pa.array(cols[7], pa.float64())}), os.path.join(DER, "prices.parquet"), compression="zstd")
    with open(os.path.join(DER, "events_yahoo.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["symbol", "ex_date", "type", "value"]); w.writerows(sorted(events))
    print(f"prices: {len(bars)} bars, {len(events)} events, {len(missing)} names without data: {missing[:10]}")


# ---- shares outstanding (Nasdaq quote summary: market cap and previous close), the funding rate, and the benchmark ----
def shares(symbols):
    os.makedirs(os.path.join(ROOT, "data", "reference"), exist_ok=True); rows = []
    for i, sym in enumerate(symbols):
        raw = get(f"https://api.nasdaq.com/api/quote/{sym.replace('.', '-')}/summary?assetclass=stocks", retries=2, timeout=30)
        if raw:
            try:
                d = json.loads(raw)["data"]["summaryData"]; mc = float(d["MarketCap"]["value"].replace(",", "")); px = float(d["PreviousClose"]["value"].replace("$", "").replace(",", ""))
                rows.append((sym, mc, px, round(mc / px) if px > 0 else "", d.get("Sector", {}).get("value", ""), d.get("Industry", {}).get("value", ""), dt.date.today().isoformat()))
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.25)
        if (i + 1) % 100 == 0:
            print(f"  shares {i + 1}/{len(symbols)}", flush=True)
    with open(os.path.join(ROOT, "data", "reference", "shares_outstanding.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["symbol", "market_cap", "price", "shares_outstanding", "sector", "industry", "as_of"]); w.writerows(rows)
    print(f"shares: {len(rows)} names")


def rates():
    os.makedirs(os.path.join(ROOT, "data", "reference"), exist_ok=True)
    raw = get("https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=2015-01-01&endDate=2030-12-31", timeout=120)
    rows = sorted((r["effectiveDate"], r["percentRate"]) for r in json.loads(raw)["refRates"])
    with open(os.path.join(ROOT, "data", "reference", "effr.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "effr"]); w.writerows(rows)
    print(f"rates: {len(rows)} days of EFFR")


if __name__ == "__main__":
    a = sys.argv[1:]; start = dt.date.fromisoformat(a[a.index("--from") + 1]) if "--from" in a else dt.date(2020, 1, 1); end = dt.date.today()
    what = [x for x in a if not x.startswith("--") and x not in ("--from",)] or ["all"]
    if "finra" in what or "all" in what:
        finra(start, end)
    if "threshold" in what or "all" in what:
        threshold(start, end)
    if "threshold-nyse" in what:
        threshold(start, end, nasdaq=False)
    if "ftd" in what or "all" in what:
        ftd(start, end)
    if "prices" in what or "all" in what:
        prices(universe() + ["SPY"])
    if "shares" in what or "all" in what:
        shares([r["symbol"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "universe.csv"), encoding="utf-8"))])
    if "rates" in what or "all" in what:
        rates()
    print("done")
