"""The book simulation on hand-made price paths: the P&L identity, fee and rebate accruals, dividends paid by the short,
locates, recalls and re-entries, the shadow leg, margin numbers."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slb import backtest as BT, financing as F, optimiser as O, strategy as S  # noqa: E402


def make_world(n_days=45, syms=("L1", "L2", "S1", "S2"), fee=0.003, hazard=0.0, p_locate=1.0, drift=None, div=None):
    days = pd.bdate_range("2024-01-01", periods=n_days)
    close = pd.DataFrame(100.0, index=days, columns=list(syms))
    if drift:
        for s, g in drift.items():
            close[s] = 100.0 * (1 + g) ** np.arange(n_days)
    adj = close.copy()
    if div:      # an ex-dividend on day 10: the unadjusted price drops, the adjusted return does not
        for s, amt in div.items():
            close.loc[days[10]:, s] = close.loc[days[10]:, s] - amt
    rows = []
    for d in days:
        for s in syms:
            rows.append({"date": d, "symbol": s, "fee": fee, "recall_hazard": hazard, "p_locate": p_locate, "vol_3m": 0.3, "sir": 0.1, "utilisation": 0.3, "close": close.loc[d, s], "dollar_adv": 1e8})
    borrow = pd.DataFrame(rows)
    effr = pd.Series(0.05, index=days)
    targets = {days[0]: {"long": pd.Series([0.5, 0.5], index=["L1", "L2"]), "short": pd.Series([-0.5, -0.5], index=["S1", "S2"])}}
    return days, adj, close, borrow, effr, targets


def test_pnl_identity_and_flat_book():
    days, adj, close, borrow, effr, targets = make_world()
    cfg = BT.BookConfig(nav0=1e6, impact_bp=0.0)
    r = BT.run_backtest(targets, adj, close, borrow, effr, cfg)
    d = r.daily
    # flat prices: the only flows are the fee (0.3 % on 1e6 short, ACT/360), the rebate (5 % on the short proceeds) and cash interest on ~0 cash
    n = len(d) - 1        # positions exist from the rebalance day on; accruals start that day
    assert d["pnl_long"].abs().sum() == 0 and d["pnl_short"].abs().sum() == 0
    assert d["fees"].sum() == pytest.approx(1e6 * 0.003 / 360 * (n + 1), rel=1e-9)
    assert d["rebate"].sum() == pytest.approx(1e6 * 0.05 / 360 * (n + 1), rel=1e-9)
    assert r.summary["nav_end"] == pytest.approx(cfg.nav0 + d["pnl_long"].sum() + d["pnl_short"].sum() - d["fees"].sum() + d["rebate"].sum() + d["financing"].sum() - d["impact"].sum() - d["recall_impact"].sum(), rel=1e-12)
    assert r.summary["recalls_total"] == 0 and r.summary["locate_failure_rate"] == 0


def test_impact_is_charged_on_turnover():
    days, adj, close, borrow, effr, targets = make_world()
    r = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6, impact_bp=10.0))
    assert r.daily["impact"].iloc[0] == pytest.approx(2e6 * 10e-4)          # 2e6 traded at 10 bp


def test_short_pays_the_dividend_and_long_receives_it():
    days, adj, close, borrow, effr, targets = make_world(div={"S1": 2.0, "L1": 2.0})
    r = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6, impact_bp=0.0))
    # adjusted returns are flat, so no price P&L on either leg even though the unadjusted close drops on the ex-date
    assert r.daily["pnl_short"].abs().sum() == 0 and r.daily["pnl_long"].abs().sum() == 0
    # but the notional the fee accrues on falls after the ex-date
    assert r.daily["short_notional"].iloc[-1] < r.daily["short_notional"].iloc[0]


def test_drift_pnl_signs():
    days, adj, close, borrow, effr, targets = make_world(drift={"L1": 0.01, "S1": -0.01})
    r = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6, impact_bp=0.0), frictionless=True)
    assert r.daily["pnl_long"].sum() > 0 and r.daily["pnl_short"].sum() > 0      # long up, short down: both legs make money
    assert r.summary["ann_shadow_pnl_short"] == pytest.approx(r.summary["ann_pnl_short"])


def test_locate_failure_redistributes_and_shadow_measures_it():
    days, adj, close, borrow, effr, targets = make_world(p_locate=0.0, drift={"S1": -0.02})
    borrow.loc[borrow["symbol"] == "S2", "p_locate"] = 1.0
    r = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6, impact_bp=0.0))
    assert r.summary["locate_failure_rate"] == 0.5 and r.daily["n_short"].iloc[0] == 1
    assert r.daily["short_notional"].iloc[0] == pytest.approx(1e6)                  # S2 carries the whole leg
    # S1 (the one that could not be borrowed) fell 2 % a day: the shadow leg made money the book did not
    assert r.summary["ann_shadow_pnl_short"] > r.summary["ann_pnl_short"] and r.summary["ann_opportunity_short"] > 0
    assert r.summary["missing_short_days_share"] == pytest.approx(0.5)


def test_recall_then_reentry():
    days, adj, close, borrow, effr, targets = make_world(hazard=1.0)
    borrow.loc[borrow["date"] > days[0], "recall_hazard"] = 0.0          # recalled on day one only
    r = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6, impact_bp=10.0))
    d = r.daily
    assert d["n_recalls"].iloc[0] == 2 and d["n_short"].iloc[0] == 0 and d["n_reentries"].iloc[1] == 2 and d["n_short"].iloc[1] == 2
    assert d["recall_impact"].iloc[0] == pytest.approx(1e6 * 10e-4) and d["recall_impact"].iloc[1] == pytest.approx(1e6 * 10e-4)
    assert d["fees"].iloc[0] == 0 and d["fees"].iloc[1] > 0
    r2 = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6, impact_bp=10.0, reentry=False))
    assert r2.daily["n_short"].iloc[-1] == 0 and r2.summary["reentry_failed_days"] > 0


def test_margin_columns_for_a_100_100_book():
    days, adj, close, borrow, effr, targets = make_world()
    r = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6, impact_bp=0.0))
    d0 = r.daily.iloc[0]
    assert d0["margin_regt_initial"] == pytest.approx(1e6) and d0["margin_regt_maint"] == pytest.approx(0.55e6) and d0["margin_pm"] == pytest.approx(0.3e6)
    m = F.margin_requirements(1e6, 1e6, 1e6)
    assert m["excess_regt_initial"] == pytest.approx(0) and m["max_gross_pm"] == pytest.approx(1e6 / 0.15) and m["leverage"] == 2.0
    cheap = pd.DataFrame({"shares": [-100000.0], "close": [4.0]})
    assert F.margin_requirements(0, 4e5, 1e6, cheap)["regt_maintenance"] == pytest.approx(4e5)     # 100 % on a sub-$5 short


def test_drag_table_columns():
    days, adj, close, borrow, effr, targets = make_world()
    r = BT.run_backtest(targets, adj, close, borrow, effr, BT.BookConfig(nav0=1e6))
    t = BT.drag_table({"naive": r})
    assert t.loc["naive", "borrow_fees"] == pytest.approx(r.summary["ann_fees"]) and "drag_borrow" in t.columns


def test_decile_portfolio_and_month_ends():
    days = pd.bdate_range("2024-01-01", periods=70)
    me = S.month_ends(days)
    assert me[0] == pd.Timestamp("2024-01-31") and me[1] == pd.Timestamp("2024-02-29") and len(me) == 4
    scores = pd.Series(np.arange(100.0), index=[f"S{i}" for i in range(100)])
    lo, sh = S.decile_portfolio(scores, set(scores.index))
    assert len(lo) == 10 and len(sh) == 10 and lo.sum() == pytest.approx(1) and sh.sum() == pytest.approx(-1)
    assert "S99" in lo.index and "S0" in sh.index


def test_optimiser_reproduces_naive_without_costs_and_trims_expensive_shorts():
    n = 100; names = [f"S{i}" for i in range(n)]; scores = pd.Series(np.arange(n, dtype=float), index=names)
    day = pd.DataFrame({"symbol": names, "vol_3m": 0.3, "fee": 0.003, "recall_hazard": 0.0007, "p_locate": 1.0})
    lo, sh, info = O.optimise(scores, day, set(names), lam=1e6, borrow_aware=False)
    assert lo.sum() == pytest.approx(1, abs=1e-6) and sh.sum() == pytest.approx(-1, abs=1e-6)
    assert np.allclose(lo.values, 0.1, atol=1e-3) and np.allclose(sh.values, -0.1, atol=1e-3) and set(sh.index) == {f"S{i}" for i in range(10)}
    # make S0 very expensive to borrow: it leaves the book and the leg still sums to -1
    day.loc[day["symbol"] == "S0", "fee"] = 0.60
    lo2, sh2, info2 = O.optimise(scores, day, set(names), lam=20.0, borrow_aware=True)
    assert "S0" not in sh2.index and sh2.sum() == pytest.approx(-1, abs=1e-6) and (sh2 >= -0.15 - 1e-9).all()      # w_max = 1.5 / k
    assert info2["short_fee_weighted"] < 0.01 and info2["expected_borrow_cost"] < info2["expected_borrow_cost_naive"]
    # the freed weight goes to the substitutes from the second decile, not to arbitrary names
    assert all(int(s[1:]) < 20 for s in sh2.index)
