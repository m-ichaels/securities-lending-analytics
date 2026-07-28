"""Daily simulation of a monthly-rebalanced long-short book with the securities-lending frictions: a locate for every new
short at the rebalance (failed locates redistribute their weight pro rata to the shorts that were found), a recall draw
on every short every day (forced cover at the close, re-entry the next day if a locate succeeds), the borrow fee accrued
on the short notional, the rebate on the short proceeds, interest on the cash balance, market impact on every crossing,
and the margin requirement under Regulation T and portfolio margin.  A frictionless shadow short leg holding the intended
shorts runs alongside, so the price return the book misses by being out of a name is measured rather than assumed."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BookConfig:
    nav0: float = 5e8
    impact_bp: float = 10.0             # one crossing, one side: liquid US names, a $500m book
    fee_day_count: int = 360
    financing_spread: float = 0.005     # over EFFR on a debit balance
    cash_spread: float = -0.001         # under EFFR on a credit balance
    regt_initial: float = 0.50
    regt_maint_long: float = 0.25
    regt_maint_short: float = 0.30
    pm_rate: float = 0.15
    reentry: bool = True
    fee_scale: float = 1.0              # sensitivity knobs
    hazard_scale: float = 1.0
    locate_scale: float = 1.0           # scales the locate failure probability (1 - p)
    seed: int = 1


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    rebalances: pd.DataFrame
    summary: dict
    config: BookConfig
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)    # date, symbol, shares, close (for the financing layer)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)       # recalls, failed locates, re-entries


def annualise(x: float, n_days: int) -> float:
    return x * 252.0 / max(n_days, 1)


def wide(borrow: pd.DataFrame, col: str) -> pd.DataFrame:
    return borrow.pivot(index="date", columns="symbol", values=col)


def _ok(v) -> bool:
    return v is not None and v == v


def run_backtest(targets: dict, adj: pd.DataFrame, close: pd.DataFrame, borrow: pd.DataFrame, effr: pd.Series, cfg: BookConfig | None = None, frictionless: bool = False, record_positions: bool = True) -> BacktestResult:
    """targets: {rebalance date: {'long': Series of weights (+), 'short': Series of weights (-)}}; adj and close are wide
    (date x symbol) frames; borrow is the long borrow table."""
    cfg = cfg or BookConfig(); rng = np.random.default_rng(cfg.seed)
    rdates = sorted(targets); days = adj.index[(adj.index >= rdates[0])]
    fee_w = wide(borrow, "fee") * cfg.fee_scale; haz_w = (wide(borrow, "recall_hazard") * cfg.hazard_scale).clip(upper=1.0); loc_w = 1.0 - (1.0 - wide(borrow, "p_locate")) * cfg.locate_scale
    effr_d = effr.reindex(days, method="ffill").bfill().fillna(0.0)
    empty = pd.Series(dtype=float)
    nav = cfg.nav0; shares: dict[str, float] = {}; shadow: dict[str, float] = {}; target_shares: dict[str, float] = {}; out_since: dict[str, pd.Timestamp] = {}
    rows = []; reb_rows = []; pos_rows = []; ev_rows = []; ri = 0; last_apx = None; last_px = None
    for d in days:
        px = close.loc[d]; apx = adj.loc[d]
        fee_row = fee_w.loc[d] if d in fee_w.index else empty; haz_row = haz_w.loc[d] if d in haz_w.index else empty; loc_row = loc_w.loc[d] if d in loc_w.index else empty
        rec = {"date": d, "nav_open": nav, "pnl_long": 0.0, "pnl_short": 0.0, "shadow_pnl_short": 0.0, "fees": 0.0, "rebate": 0.0, "financing": 0.0, "impact": 0.0, "recall_impact": 0.0, "n_recalls": 0, "n_reentries": 0, "n_reentry_failed": 0, "n_missing": 0, "n_short": 0, "n_long": 0, "short_notional": 0.0, "long_notional": 0.0, "fee_wavg": 0.0}
        # ---- mark to market: adjusted-close returns on the unadjusted notional (the short pays the dividend) ------------------------
        if last_apx is not None:
            for s, q in shares.items():
                a0 = last_apx.get(s); a1 = apx.get(s); p0 = last_px.get(s)
                if _ok(a0) and _ok(a1) and _ok(p0) and a0 > 0:
                    p = q * (a1 / a0 - 1.0) * p0
                    if q > 0:
                        rec["pnl_long"] += p
                    else:
                        rec["pnl_short"] += p
            for s, q in shadow.items():
                a0 = last_apx.get(s); a1 = apx.get(s); p0 = last_px.get(s)
                if _ok(a0) and _ok(a1) and _ok(p0) and a0 > 0:
                    rec["shadow_pnl_short"] += q * (a1 / a0 - 1.0) * p0
        nav += rec["pnl_long"] + rec["pnl_short"]
        # ---- rebalance at the month-end close ---------------------------------------------------------------------------------------
        if ri < len(rdates) and d == rdates[ri]:
            t = targets[d]; ri += 1; new_long, new_short = t["long"], t["short"]
            target_shares = {}
            for s, w in new_long.items():
                p = px.get(s)
                if _ok(p) and p > 0:
                    target_shares[s] = w * nav / p
            short_ok = {}; failed = []; n_req = 0
            for s, w in new_short.items():
                p = px.get(s)
                if not (_ok(p) and p > 0):
                    continue
                n_req += 1; pl = loc_row.get(s, 1.0); pl = 1.0 if frictionless or not _ok(pl) else float(pl)
                if shares.get(s, 0.0) < 0 or rng.random() < pl:
                    short_ok[s] = w
                else:
                    failed.append(s); ev_rows.append({"date": d, "symbol": s, "event": "locate_failed", "p_locate": pl})
            tot = sum(short_ok.values())
            if short_ok and tot != 0:
                scale = new_short.sum() / tot
                for s, w in short_ok.items():
                    target_shares[s] = w * scale * nav / px[s]
            shadow = {s: w * nav / px[s] for s, w in new_short.items() if _ok(px.get(s)) and px[s] > 0}
            turnover = 0.0
            for s in set(shares) | set(target_shares):
                p = px.get(s)
                if _ok(p):
                    turnover += abs(target_shares.get(s, 0.0) - shares.get(s, 0.0)) * p
            rec["impact"] += turnover * cfg.impact_bp * 1e-4; nav -= turnover * cfg.impact_bp * 1e-4
            shares = {s: q for s, q in target_shares.items() if abs(q) > 0}; out_since = {}
            wts = [abs(w) for w in short_ok.values()]
            reb_rows.append({"date": d, "n_long": len(new_long), "n_short_requested": n_req, "n_locate_failed": len(failed), "failed": ",".join(failed[:12]), "turnover_usd": turnover, "nav": nav,
                             "fee_wavg_short": float(np.average([fee_row.get(s, 0.003) for s in short_ok], weights=wts)) if wts else 0.0, "short_leg_fee_requested": float(np.average([fee_row.get(s, 0.003) if _ok(fee_row.get(s)) else 0.003 for s in new_short.index], weights=[abs(w) for w in new_short.values])) if len(new_short) else 0.0})
        # ---- recalls and re-entries -------------------------------------------------------------------------------------------------
        if not frictionless:
            for s in [s for s, q in shares.items() if q < 0]:
                h = haz_row.get(s, 0.001); h = h if _ok(h) else 0.001; p = px.get(s)
                if _ok(p) and rng.random() < h:
                    q = shares.pop(s); cost = abs(q) * p * cfg.impact_bp * 1e-4; rec["recall_impact"] += cost; nav -= cost; rec["n_recalls"] += 1; out_since[s] = d
                    ev_rows.append({"date": d, "symbol": s, "event": "recall", "notional": abs(q) * p, "hazard": h})
            for s in list(out_since):
                if out_since[s] < d and s in target_shares and s not in shares:
                    p = px.get(s); pl = loc_row.get(s, 1.0); pl = pl if _ok(pl) else 1.0
                    if cfg.reentry and _ok(p) and p > 0 and rng.random() < pl:
                        q = target_shares[s]; shares[s] = q; cost = abs(q) * p * cfg.impact_bp * 1e-4; rec["recall_impact"] += cost; nav -= cost; rec["n_reentries"] += 1; out_since.pop(s)
                        ev_rows.append({"date": d, "symbol": s, "event": "reentry", "notional": abs(q) * p})
                    else:
                        rec["n_reentry_failed"] += 1
        # ---- accruals ------------------------------------------------------------------------------------------------------------------
        long_notional = 0.0; short_notional = 0.0; fees = 0.0; wsum = 0.0
        for s, q in shares.items():
            p = px.get(s)
            if not _ok(p):
                continue
            if q > 0:
                long_notional += q * p
            else:
                n = -q * p; short_notional += n
                if not frictionless:
                    f = fee_row.get(s, 0.003); f = f if _ok(f) else 0.003; fees += n * f / cfg.fee_day_count; wsum += n * f
            if record_positions:
                pos_rows.append((d, s, q, p))
        rec["fees"] = fees; nav -= fees; rec["fee_wavg"] = wsum / short_notional if short_notional > 0 else 0.0
        r = float(effr_d.loc[d]); rec["rebate"] = short_notional * r / 360.0
        cash = nav - long_notional; rec["financing"] = cash * ((r + cfg.financing_spread) if cash < 0 else (r + cfg.cash_spread)) / 360.0
        nav += rec["rebate"] + rec["financing"]
        rec["n_missing"] = sum(1 for s in shadow if s not in shares)
        rec["n_short"] = sum(1 for q in shares.values() if q < 0); rec["n_long"] = sum(1 for q in shares.values() if q > 0)
        rec["short_notional"] = short_notional; rec["long_notional"] = long_notional; rec["cash"] = cash
        rec["margin_regt_initial"] = cfg.regt_initial * (long_notional + short_notional); rec["margin_regt_maint"] = cfg.regt_maint_long * long_notional + cfg.regt_maint_short * short_notional; rec["margin_pm"] = cfg.pm_rate * (long_notional + short_notional)
        rec["nav"] = nav; rows.append(rec); last_apx = apx; last_px = px
    daily = pd.DataFrame(rows).set_index("date"); reb = pd.DataFrame(reb_rows)
    n = len(daily); nav0 = cfg.nav0
    daily["excess_equity_regt"] = daily["nav"] - daily["margin_regt_maint"]; daily["excess_equity_pm"] = daily["nav"] - daily["margin_pm"]
    ann = lambda col: annualise(float((daily[col] / daily["nav_open"]).sum()), n)       # flows as a fraction of that day's NAV, so they read as return contributions
    summ = {"days": n, "months": int(len(reb)), "nav_end": float(nav), "total_return": float(nav / nav0 - 1), "ann_return": annualise(float(np.log(nav / nav0)), n), "ann_return_simple": ann("pnl_long") + ann("pnl_short") - ann("fees") + ann("rebate") + ann("financing") - ann("impact") - ann("recall_impact"),
            "ann_pnl_long": ann("pnl_long"), "ann_pnl_short": ann("pnl_short"), "ann_shadow_pnl_short": ann("shadow_pnl_short"), "ann_fees": ann("fees"), "ann_rebate": ann("rebate"), "ann_financing": ann("financing"), "ann_impact": ann("impact"), "ann_recall_impact": ann("recall_impact"),
            "ann_opportunity_short": ann("shadow_pnl_short") - ann("pnl_short"), "recalls_total": int(daily["n_recalls"].sum()), "recalls_per_month": float(daily["n_recalls"].sum() / max(len(reb), 1)), "reentries": int(daily["n_reentries"].sum()), "reentry_failed_days": int(daily["n_reentry_failed"].sum()),
            "locate_failure_rate": float(reb["n_locate_failed"].sum() / max(reb["n_short_requested"].sum(), 1)) if len(reb) else 0.0, "missing_short_days_share": float(daily["n_missing"].sum() / max((daily["n_short"] + daily["n_missing"]).sum(), 1)),
            "fee_wavg_short": float(np.average(daily["fee_wavg"], weights=daily["short_notional"].clip(lower=1))) if n else 0.0, "fee_wavg_requested": float(reb["short_leg_fee_requested"].mean()) if len(reb) else 0.0,
            "ann_vol": float(np.log(daily["nav"]).diff().std() * np.sqrt(252)), "max_drawdown": float((daily["nav"] / daily["nav"].cummax() - 1).min()), "avg_gross_over_nav": float(((daily["long_notional"] + daily["short_notional"]) / daily["nav"]).mean()),
            "margin_breaches_regt_maint": int((daily["excess_equity_regt"] < 0).sum()), "margin_breaches_pm": int((daily["excess_equity_pm"] < 0).sum()), "min_excess_pm_over_nav": float((daily["excess_equity_pm"] / daily["nav"]).min()), "min_excess_regt_over_nav": float((daily["excess_equity_regt"] / daily["nav"]).min()),
            "avg_turnover_over_nav": float((reb["turnover_usd"] / reb["nav"]).mean()) if len(reb) else 0.0}
    summ["ann_drag_borrow"] = summ["ann_fees"] + summ["ann_recall_impact"] + summ["ann_opportunity_short"]
    summ["ann_drag_certain"] = summ["ann_fees"] + summ["ann_recall_impact"]
    summ["recalls_per_100_position_months"] = 100.0 * summ["recalls_total"] / max(float(daily["n_short"].sum()) / 21.0, 1.0)
    summ["avg_n_short"] = float(daily["n_short"].mean()); summ["avg_n_long"] = float(daily["n_long"].mean())
    summ["ann_sharpe"] = summ["ann_return"] / summ["ann_vol"] if summ["ann_vol"] > 0 else 0.0
    pos = pd.DataFrame(pos_rows, columns=["date", "symbol", "shares", "close"]) if pos_rows else pd.DataFrame(columns=["date", "symbol", "shares", "close"])
    return BacktestResult(daily, reb, summ, cfg, pos, pd.DataFrame(ev_rows))


def drag_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """one row per book: the annualised decomposition from gross price P&L to net"""
    rows = []
    for k, r in results.items():
        s = r.summary
        rows.append({"book": k, "gross_price_pnl": s["ann_pnl_long"] + s["ann_pnl_short"], "pnl_long": s["ann_pnl_long"], "pnl_short": s["ann_pnl_short"], "shadow_pnl_short": s["ann_shadow_pnl_short"], "opportunity_cost": s["ann_opportunity_short"], "borrow_fees": s["ann_fees"], "recall_impact": s["ann_recall_impact"], "rebalance_impact": s["ann_impact"],
                     "rebate": s["ann_rebate"], "cash_financing": s["ann_financing"], "net_return": s["ann_return"], "drag_borrow": s["ann_drag_borrow"], "drag_certain": s["ann_drag_certain"], "fee_wavg": s["fee_wavg_short"], "locate_failure_rate": s["locate_failure_rate"], "recalls_per_month": s["recalls_per_month"], "recalls_per_100_position_months": s["recalls_per_100_position_months"], "missing_short_days_share": s["missing_short_days_share"], "avg_n_short": s["avg_n_short"], "ann_vol": s["ann_vol"], "sharpe": s["ann_sharpe"]})
    return pd.DataFrame(rows).set_index("book")
