"""The borrow model on a synthetic panel: publication lags, split-adjusted shares, the score ordering, the fee map, the
isotonic fit, supply and hazard bounds."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slb import borrow as B  # noqa: E402


def make_prices(symbols, start="2024-01-01", n=120, seed=0):
    rng = np.random.default_rng(seed); days = pd.bdate_range(start, periods=n)
    rows = []
    for k, s in enumerate(symbols):
        px = 20.0 * (1 + 0.02 * k) * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for d, p in zip(days, px):
            rows.append((s, d, p, p, p, p, p, 1e6 * (1 + k)))
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "adjclose", "volume"])


@pytest.fixture
def world():
    syms = ["AAA", "BBB", "CCC", "DDD"]
    px = make_prices(syms)
    sec = pd.DataFrame({"symbol": syms, "shares_outstanding": [100e6, 50e6, 10e6, 200e6], "market_cap": [5e9, 1e9, 2e8, 4e10]})
    si = pd.DataFrame([("2024-01-31", "AAA", 2e6, 1.5e6, 1e6, 2.0, "NYSE"), ("2024-01-31", "BBB", 10e6, 9e6, 2e6, 5.0, "NNM"), ("2024-01-31", "CCC", 4e6, 4e6, 3e6, 1.3, "NNM"), ("2024-01-31", "DDD", 1e6, 1e6, 4e6, 0.25, "NYSE"),
                       ("2024-02-15", "CCC", 6e6, 4e6, 3e6, 2.0, "NNM")], columns=["settle_date", "symbol", "shares_short", "prev_short", "adv", "days_to_cover", "market"])
    si["settle_date"] = pd.to_datetime(si["settle_date"])
    th = pd.DataFrame({"date": pd.to_datetime(["2024-02-05", "2024-02-06"]), "symbol": ["CCC", "CCC"], "market": ["Q", "Q"], "source": ["nasdaq", "nasdaq"]})
    ftd = pd.DataFrame({"settle_date": pd.to_datetime(["2024-01-20", "2024-01-22"]), "cusip": ["c", "c"], "symbol": ["CCC", "CCC"], "quantity": [300000.0, 400000.0], "price": [20.0, 21.0]})
    events = pd.DataFrame({"symbol": ["DDD"], "ex_date": ["2024-03-01"], "type": ["split"], "value": [2.0], "ratio_text": ["2:1"], "note": [""]})
    return syms, px, sec, si, th, ftd, events


def test_publication_lag_and_forward_fill(world):
    syms, px, sec, si, th, ftd, events = world
    panel = B.daily_panel(px, si, th, ftd, sec, events, start="2024-01-01")
    a = panel[panel["symbol"] == "AAA"].set_index("date")
    days = pd.DatetimeIndex(sorted(px["date"].unique()))
    pub = B.business_days_after(pd.Timestamp("2024-01-31"), B.PUBLICATION_LAG_BD, days)
    assert pub == days[days.searchsorted(pd.Timestamp("2024-01-31")) + 9]
    assert np.isnan(a.loc[pub - pd.Timedelta(days=1), "shares_short"]) and a.loc[pub, "shares_short"] == 2e6
    assert a.loc[days[-1], "shares_short"] == 2e6                                 # forward-filled to the end
    assert abs(a.loc[pub, "sir"] - 0.02) < 1e-12


def test_threshold_next_day_with_persistence(world):
    syms, px, sec, si, th, ftd, events = world
    panel = B.daily_panel(px, si, th, ftd, sec, events, start="2024-01-01")
    c = panel[panel["symbol"] == "CCC"].set_index("date")["on_threshold"]
    assert not c.loc["2024-02-05"] and c.loc["2024-02-06"] and c.loc["2024-02-07"]
    assert c.loc["2024-02-13"] and not c.loc["2024-02-14"]            # five business days after the last listing


def test_split_adjusted_shares_outstanding(world):
    syms, px, sec, si, th, ftd, events = world
    panel = B.daily_panel(px, si, th, ftd, sec, events, start="2024-01-01")
    d = panel[panel["symbol"] == "DDD"].set_index("date")["shares_out"]
    assert d.loc["2024-02-28"] == 100e6 and d.loc["2024-03-01"] == 200e6


def test_ftd_lag(world):
    syms, px, sec, si, th, ftd, events = world
    panel = B.daily_panel(px, si, th, ftd, sec, events, start="2024-01-01")
    c = panel[panel["symbol"] == "CCC"].set_index("date")
    first = pd.Timestamp("2024-01-20") + pd.Timedelta(days=B.FTD_LAG_DAYS)
    before = c.loc[:first - pd.Timedelta(days=1), "ftd_ratio"]; after = c.loc[first:, "ftd_ratio"]
    assert (before == 0).all() and (after > 0).all()


def test_score_orders_the_special_name_first(world):
    syms, px, sec, si, th, ftd, events = world
    panel = B.daily_panel(px, si, th, ftd, sec, events, start="2024-01-01")
    borrow, info = B.build_borrow_table(panel)
    last = borrow[borrow["date"] == borrow["date"].max()].set_index("symbol")
    assert last["score"].idxmax() == "CCC" and last["fee"].idxmax() == "CCC" and last["fee"].idxmin() == "DDD"
    assert info["source"] == "quantile_map"
    assert (borrow["fee"] >= B.GC_FEE - 1e-12).all() and (borrow["p_locate"].between(0.05, 1.0)).all() and (borrow["recall_hazard"] > 0).all()


def test_fee_map_hits_the_quantiles():
    xs = np.array([q for q, _ in B.FEE_QUANTILES]); ys = np.array([f for _, f in B.FEE_QUANTILES])
    assert np.allclose(B.fee_from_percentile(xs), ys)
    p = np.linspace(0, 1, 101); f = B.fee_from_percentile(p)
    assert (np.diff(f) >= -1e-12).all() and abs(f[50] - 0.003) < 1e-9 and f[-1] == 1.0


def test_pav_is_monotone_least_squares():
    y = np.array([1.0, 3.0, 2.0, 4.0, 3.5, 6.0])
    fit = B.pav(y)
    assert (np.diff(fit) >= 0).all() and np.allclose(fit, [1.0, 2.5, 2.5, 3.75, 3.75, 6.0])


def test_isotonic_fit_used_when_ib_snapshots_exist(world):
    syms, px, sec, si, th, ftd, events = world
    panel = B.daily_panel(px, si, th, ftd, sec, events, start="2024-01-01")
    panel["score"] = B.specialness_score(panel)
    # an IB file that says fees rise with the score, 300 pairs
    ib = panel[["date", "symbol", "score"]].tail(300).copy(); ib["fee_rate"] = 100 * (0.003 + 0.05 * np.exp(ib["score"]))
    fee, info = B.fee_from_score(panel, ib)
    assert info["source"] == "ib_isotonic" and info["spearman"] > 0.99 and info["n_pairs"] == 300
    last = panel[panel["date"] == panel["date"].max()].index
    f = fee.loc[last]; sc = panel.loc[last, "score"]
    assert f[sc.idxmax()] >= f[sc.idxmin()]


def test_supply_and_hazard_shapes():
    base = pd.DataFrame({"shares_out": [100e6] * 4, "cap": ["large"] * 4, "shares_short": [6e6, 18e6, 30e6, 48e6], "on_threshold": [False, False, False, True]})
    out = B.supply_and_hazard(base)
    assert list(out["lendable"]) == [30e6] * 4
    assert np.allclose(out["utilisation"], [0.2, 0.6, 1.0, 1.5])
    assert out["p_locate"].iloc[0] == 1.0 and out["p_locate"].iloc[1] < 1.0 and out["p_locate"].iloc[2] == pytest.approx(0.3) and out["p_locate"].iloc[3] == pytest.approx(0.05)
    assert (np.diff(out["recall_hazard"]) > 0).all()


def test_reverse_split_after_settlement_keeps_the_ratio(world):
    syms, px, sec, si, th, ftd, events = world
    # DDD: short interest settled 31 Jan (1e6 of 100e6 pre-split shares = 1 %); a 1:2 reverse split on 1 Mar halves the share count
    ev = pd.DataFrame({"symbol": ["DDD"], "ex_date": ["2024-03-01"], "type": ["split"], "value": [0.5], "ratio_text": ["1:2"], "note": [""]})
    sec2 = sec.copy(); sec2.loc[sec2["symbol"] == "DDD", "shares_outstanding"] = 100e6      # today's (post-split) count
    panel = B.daily_panel(px, si, th, ftd, sec2, ev, start="2024-01-01")
    d = panel[panel["symbol"] == "DDD"].set_index("date")
    assert d.loc["2024-02-28", "shares_out"] == 200e6 and d.loc["2024-03-01", "shares_out"] == 100e6
    assert d.loc["2024-02-28", "sir"] == pytest.approx(1e6 / 200e6) and d.loc["2024-03-15", "sir"] == pytest.approx(1e6 / 200e6)
    assert d.loc["2024-03-15", "shares_short"] == pytest.approx(0.5e6)              # the count in today's shares


def test_fee_floor_from_utilisation():
    f = B.fee_floor(np.array([0.0, 0.3, 0.7, 1.0, 1.2, 1.5]))
    assert np.allclose(f, [0.0025, 0.0025, 0.0025, 0.025, 0.075, 0.20]) and (np.diff(f) >= -1e-12).all()
    base = pd.DataFrame({"shares_out": [100e6] * 3, "cap": ["large"] * 3, "shares_short": [5e6, 30e6, 36e6], "on_threshold": [False] * 3, "fee": [0.003, 0.003, 0.003]})
    out = B.supply_and_hazard(base)
    assert list(out["fee_floor_binds"]) == [False, True, True] and out["fee"].iloc[1] == pytest.approx(0.025) and out["fee"].iloc[2] == pytest.approx(B.fee_floor(np.array([1.2]))[0])
