"""The pipeline: store -> borrow table -> three long-short books under four treatments each -> drag decomposition, the
optimiser's recovery, sensitivities -> the financing layer and the prime-broker reconciliation on the headline book ->
results/*.json and the daily series the figures are drawn from."""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd

from . import backtest as BT, borrow as B, data as D, financing as F, optimiser as O, strategy as S

RESULTS = os.path.join(D.ROOT, "results")
BORROW_PATH = os.path.join(D.DER, "borrow.parquet")
HEADLINE = "MOM"
LAM = 20.0


def to_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (pd.Timestamp, dt.date, dt.datetime)):
            return str(o)[:10]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        return str(o)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=default)


# ---- borrow table ---------------------------------------------------------------------------------------------------------------
def build_borrow(con=None, start: str = "2019-06-01", verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    own = con is None
    con = con or D.connect()
    prices = con.execute("select symbol, date, open, high, low, close, adjclose, volume from prices order by symbol, date").df()
    si = con.execute("select * from short_interest").df(); th = con.execute("select * from threshold").df(); ftd = con.execute("select * from ftd").df()
    sec = con.execute("select * from securities").df(); ib = con.execute("select * from ib_snapshots").df()
    events = D.load_events()
    for c, df in (("settle_date", si), ("date", th), ("settle_date", ftd), ("date", ib)):
        if len(df):
            df[c] = pd.to_datetime(df[c])
    t0 = time.time()
    panel = B.daily_panel(prices, si, th, ftd, sec, events, start=start)
    borrow, info = B.build_borrow_table(panel, ib if len(ib) else None)
    info["rows"] = int(len(borrow)); info["symbols"] = int(borrow["symbol"].nunique()); info["days"] = int(borrow["date"].nunique()); info["from"] = str(borrow["date"].min().date()); info["to"] = str(borrow["date"].max().date()); info["seconds"] = round(time.time() - t0, 1)
    info["inputs"] = {"short_interest_rows": int(len(si)), "threshold_rows": int(len(th)), "threshold_days": int(th["date"].nunique()) if len(th) else 0, "ftd_rows": int(len(ftd)), "ib_rows": int(len(ib)), "with_shares_outstanding": int(sec["shares_outstanding"].notna().sum())}
    borrow.to_parquet(BORROW_PATH, index=False)
    con.execute("delete from borrow")
    con.execute("insert into borrow select cast(date as date), symbol, sir, days_to_cover, on_threshold, ftd_ratio, score, fee, fee_floor_binds, lendable, utilisation, p_locate, recall_hazard, source from borrow")
    if own:
        con.close()
    if verbose:
        print({k: v for k, v in info.items() if k != "quantiles"})
    return borrow, info


def load_borrow(start: str | None = None) -> pd.DataFrame:
    b = pd.read_parquet(BORROW_PATH); b["date"] = pd.to_datetime(b["date"])
    return b[b["date"] >= start] if start else b


def wide_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = prices.copy(); p["date"] = pd.to_datetime(p["date"])
    adj = p.pivot(index="date", columns="symbol", values="adjclose").sort_index(); close = p.pivot(index="date", columns="symbol", values="close").sort_index()
    return adj, close


# ---- checks on the borrow model -------------------------------------------------------------------------------------------------
def fee_checks(borrow: pd.DataFrame, adj: pd.DataFrame) -> dict:
    """the cross-section against the published one, the threshold names, and the forward return by fee bucket (the
    literature's 'expensive to short means low returns') as a sign the ordering carries information"""
    last = borrow[borrow["date"] == borrow["date"].max()]
    dist = B.fee_distribution(borrow)
    monthly = borrow.groupby(borrow["date"].dt.to_period("M")).agg(mean_fee=("fee", "mean"), p90_fee=("fee", lambda s: s.quantile(0.9)), share_special=("fee", lambda s: float((s > 0.05).mean())), share_floor=("fee_floor_binds", "mean"), on_threshold=("on_threshold", "sum"), mean_sir=("sir", "mean"), mean_util=("utilisation", "mean"), n=("symbol", "nunique"))
    monthly.index = monthly.index.astype(str)
    thr = borrow[borrow["on_threshold"]]["fee"]; nthr = borrow[~borrow["on_threshold"]]["fee"]
    # forward one-month return by fee bucket at each month end
    rdates = S.month_ends(pd.DatetimeIndex(sorted(borrow["date"].unique())))
    rows = []
    for i in range(len(rdates) - 1):
        d0, d1 = rdates[i], rdates[i + 1]
        day = borrow[borrow["date"] == d0].set_index("symbol")
        day = day[(day["close"] >= 5) & (day["dollar_adv"] >= 5e6)]
        r = (adj.loc[d1] / adj.loc[d0] - 1).reindex(day.index)
        bucket = pd.cut(day["fee"], [0, 0.01, 0.05, 10], labels=["gc", "1-5%", ">5%"])
        for b_, g in r.groupby(bucket, observed=True):
            rows.append({"date": d0, "bucket": str(b_), "ret": float(g.mean()), "n": int(g.notna().sum())})
    fr = pd.DataFrame(rows)
    fwd = {}
    if len(fr):
        for b_, g in fr.groupby("bucket"):
            fwd[b_] = {"mean_monthly_return": float(g["ret"].mean()), "months": int(len(g)), "avg_names": float(g["n"].mean()), "t_stat": float(g["ret"].mean() / (g["ret"].std() / np.sqrt(len(g)))) if len(g) > 2 and g["ret"].std() > 0 else None}
        piv = fr.pivot(index="date", columns="bucket", values="ret")
        if ">5%" in piv and "gc" in piv:
            sp = (piv["gc"] - piv[">5%"]).dropna(); fwd["gc_minus_special"] = {"mean_monthly": float(sp.mean()), "t_stat": float(sp.mean() / (sp.std() / np.sqrt(len(sp)))) if len(sp) > 2 else None, "months": int(len(sp))}
    return {"distribution_last_day": dist, "literature_targets": {"share_gc_below_1pct": "0.85-0.92 (D'Avolio 2002: 91% of names GC; Engelberg et al 2018)", "mean_fee": "0.5-1.5%", "p99": "0.20-0.50"},
            "threshold_names": {"mean_fee_on_threshold": float(thr.mean()) if len(thr) else None, "mean_fee_off_threshold": float(nthr.mean()), "n_days_on": int(len(thr))},
            "monthly": monthly.round(5).to_dict(orient="index"), "forward_return_by_fee_bucket": fwd, "fee_source": borrow["source"].iloc[0]}


# ---- strategies ------------------------------------------------------------------------------------------------------------------
def run_strategy(name: str, borrow: pd.DataFrame, adj: pd.DataFrame, close: pd.DataFrame, effr: pd.Series, cfg: BT.BookConfig | None = None, lam: float = LAM, alpha_spread: float = O.ALPHA_MONTHLY_SPREAD, books: tuple = ("frictionless", "naive", "risk_only", "borrow_aware"), record_positions: bool = True) -> dict:
    cfg = cfg or BT.BookConfig()
    rdates = S.month_ends(pd.DatetimeIndex(sorted(borrow["date"].unique())))[:-1]
    t0 = time.time(); naive = S.naive_targets(adj, borrow, rdates, name)
    if not naive:
        return {"error": "no rebalances"}
    targets = {"naive": naive, "frictionless": naive}
    if "risk_only" in books:
        targets["risk_only"] = O.optimised_targets(naive, borrow, borrow_aware=False, lam=lam, alpha_spread=alpha_spread)
    if "borrow_aware" in books:
        targets["borrow_aware"] = O.optimised_targets(naive, borrow, borrow_aware=True, lam=lam, alpha_spread=alpha_spread)
    results = {}
    for bk in books:
        results[bk] = BT.run_backtest(targets[bk], adj, close, borrow, effr, cfg, frictionless=(bk == "frictionless"), record_positions=record_positions and bk in ("naive", "borrow_aware"))
    table = BT.drag_table(results)
    out = {"signal": S.SIGNALS[name], "rebalances": len(naive), "first": str(min(naive).date()), "last": str(max(naive).date()), "seconds": round(time.time() - t0, 1), "books": {k: r.summary for k, r in results.items()}, "drag_table": table.to_dict(orient="index"),
           "optimiser": {bk: {"avg_expected_borrow_cost_monthly": float(np.mean([t["info"]["expected_borrow_cost"] for t in targets[bk].values()])), "avg_short_fee_weighted": float(np.mean([t["info"]["short_fee_weighted"] for t in targets[bk].values()])), "avg_n_short": float(np.mean([t["info"]["n_short"] for t in targets[bk].values()])), "avg_short_in_bottom_decile": float(np.mean([t["info"]["short_in_bottom_decile"] for t in targets[bk].values()])), "avg_active_share": float(np.mean([t["info"]["active_share"] for t in targets[bk].values()])), "avg_expected_borrow_cost_naive": float(np.mean([t["info"]["expected_borrow_cost_naive"] for t in targets[bk].values()])), "success_rate": float(np.mean([t["info"]["success"] for t in targets[bk].values()]))} for bk in ("risk_only", "borrow_aware") if bk in targets},
           "lam": lam, "alpha_spread_monthly": alpha_spread}
    if "naive" in results and "borrow_aware" in results and "risk_only" in results:
        n, a, r = results["naive"].summary, results["borrow_aware"].summary, results["risk_only"].summary
        out["recovery"] = {"drag_naive": n["ann_drag_borrow"], "drag_risk_only": r["ann_drag_borrow"], "drag_borrow_aware": a["ann_drag_borrow"], "drag_reduction_vs_risk_only": r["ann_drag_borrow"] - a["ann_drag_borrow"], "drag_reduction_vs_naive": n["ann_drag_borrow"] - a["ann_drag_borrow"],
                           "net_return_naive": n["ann_return"], "net_return_risk_only": r["ann_return"], "net_return_borrow_aware": a["ann_return"], "net_gain_vs_risk_only": a["ann_return"] - r["ann_return"], "net_gain_vs_naive": a["ann_return"] - n["ann_return"],
                           "fees_naive": n["ann_fees"], "fees_borrow_aware": a["ann_fees"], "recalls_per_month_naive": n["recalls_per_month"], "recalls_per_month_borrow_aware": a["recalls_per_month"], "locate_failure_naive": n["locate_failure_rate"], "locate_failure_borrow_aware": a["locate_failure_rate"],
                           "certain_naive": n["ann_drag_certain"], "certain_risk_only": r["ann_drag_certain"], "certain_borrow_aware": a["ann_drag_certain"], "certain_recovery_vs_risk_only": r["ann_drag_certain"] - a["ann_drag_certain"], "certain_recovery_vs_naive": n["ann_drag_certain"] - a["ann_drag_certain"]}
    return {"summary": out, "results": results, "targets": targets}


def sensitivity(name: str, borrow: pd.DataFrame, adj: pd.DataFrame, close: pd.DataFrame, effr: pd.Series, base: BT.BookConfig | None = None, lam: float = LAM, grid=((0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 0.5), (1.0, 1.5), (1.5, 1.5), (2.0, 2.0)), seeds=(1, 2, 3)) -> dict:
    """fee and hazard scales on the naive and the borrow-aware book; seeds on the base case"""
    base = base or BT.BookConfig()
    rdates = S.month_ends(pd.DatetimeIndex(sorted(borrow["date"].unique())))[:-1]
    naive = S.naive_targets(adj, borrow, rdates, name)
    out = {"grid": [], "seeds": []}
    for fs, hs in grid:
        b = borrow.copy(); b["fee"] = b["fee"] * fs; b["recall_hazard"] = (b["recall_hazard"] * hs).clip(upper=0.5)
        aware = O.optimised_targets(naive, b, borrow_aware=True, lam=lam)
        cfg = BT.BookConfig(**{**base.__dict__, "fee_scale": 1.0, "hazard_scale": 1.0})
        rn = BT.run_backtest(naive, adj, close, b, effr, cfg, record_positions=False); ra = BT.run_backtest(aware, adj, close, b, effr, cfg, record_positions=False)
        out["grid"].append({"fee_scale": fs, "hazard_scale": hs, "drag_naive": rn.summary["ann_drag_borrow"], "drag_borrow_aware": ra.summary["ann_drag_borrow"], "certain_naive": rn.summary["ann_drag_certain"], "certain_aware": ra.summary["ann_drag_certain"], "net_naive": rn.summary["ann_return"], "net_borrow_aware": ra.summary["ann_return"], "fees_naive": rn.summary["ann_fees"], "fees_aware": ra.summary["ann_fees"], "recalls_naive": rn.summary["recalls_per_month"], "recalls_aware": ra.summary["recalls_per_month"], "recall_impact_naive": rn.summary["ann_recall_impact"], "recall_impact_aware": ra.summary["ann_recall_impact"], "gain": ra.summary["ann_return"] - rn.summary["ann_return"]})
    aware = O.optimised_targets(naive, borrow, borrow_aware=True, lam=lam)
    for sd in seeds:
        cfg = BT.BookConfig(**{**base.__dict__, "seed": sd})
        rn = BT.run_backtest(naive, adj, close, borrow, effr, cfg, record_positions=False); ra = BT.run_backtest(aware, adj, close, borrow, effr, cfg, record_positions=False)
        out["seeds"].append({"seed": sd, "drag_naive": rn.summary["ann_drag_borrow"], "drag_borrow_aware": ra.summary["ann_drag_borrow"], "certain_naive": rn.summary["ann_drag_certain"], "certain_aware": ra.summary["ann_drag_certain"], "net_naive": rn.summary["ann_return"], "net_borrow_aware": ra.summary["ann_return"], "recalls_naive": rn.summary["recalls_per_month"], "opportunity_naive": rn.summary["ann_opportunity_short"], "locate_failure_naive": rn.summary["locate_failure_rate"]})
    return out


# ---- financing layer -------------------------------------------------------------------------------------------------------------
def run_financing(res: BT.BacktestResult, borrow: pd.DataFrame, effr: pd.Series, con=None, renames: pd.DataFrame | None = None, book: str = "LS", tol_usd: float = 0.5) -> dict:
    fee_w = BT.wide(borrow, "fee")
    acc = F.book_accruals(res.positions, res.daily, fee_w, effr, book=book)
    pb = F.pb_statement(acc, res.positions, res.events, renames)
    breaks, score = F.reconcile(acc, pb, renames, tol_usd=tol_usd)
    if con is not None:
        con.execute("delete from accruals"); con.execute("insert into accruals select cast(date as date), book, symbol, side, notional, rate, day_count, amount, kind from acc")
        pbs = pb.drop(columns=["seeded"]); con.execute("delete from pb_statement"); con.execute("insert into pb_statement select cast(date as date), book, line_id, symbol, side, notional, rate, day_count, amount, kind from pbs")
        br = breaks.copy(); br["book"] = book; br["key"] = br["date"].astype(str) + "|" + br["symbol"] + "|" + br["kind"]
        con.execute("delete from fin_breaks"); con.execute("insert into fin_breaks select cast(date as date), book, break_type, symbol, key, detail, seeded from br")
    os.makedirs(RESULTS, exist_ok=True); breaks.to_csv(os.path.join(RESULTS, "fin_breaks.csv"), index=False)
    d = res.daily; last = d.iloc[-1]
    margin = F.margin_requirements(float(last["long_notional"]), float(last["short_notional"]), float(last["nav"]), res.positions[(res.positions["date"] == d.index[-1]) & (res.positions["shares"] < 0)])
    fs = F.financing_summary(acc, d["nav_open"], len(d))
    fs["margin_last_day"] = margin; fs["margin_breaches_regt_maintenance"] = res.summary["margin_breaches_regt_maint"]; fs["margin_breaches_pm"] = res.summary["margin_breaches_pm"]; fs["min_excess_pm_over_nav"] = res.summary["min_excess_pm_over_nav"]; fs["min_excess_regt_over_nav"] = res.summary["min_excess_regt_over_nav"]
    fs["reconciliation"] = score
    return fs


# ---- everything --------------------------------------------------------------------------------------------------------------------
def run_all(quick: bool = False, start: str = "2019-06-01", names: tuple = ("MOM", "SIR", "REV"), verbose: bool = True, reuse: bool = False) -> dict:
    t0 = time.time(); os.makedirs(RESULTS, exist_ok=True)
    con = D.connect()
    if reuse and os.path.exists(BORROW_PATH) and con.execute("select count(*) from prices").fetchone()[0] > 0:
        counts = D.store_counts(con); counts["reused"] = True; borrow = load_borrow()
        binfo = {"source": str(borrow["source"].iloc[0]), "rows": int(len(borrow)), "symbols": int(borrow["symbol"].nunique()), "days": int(borrow["date"].nunique()), "from": str(borrow["date"].min().date()), "to": str(borrow["date"].max().date()), "fee_floor_binds_share": float(borrow["fee_floor_binds"].mean()) if "fee_floor_binds" in borrow else None, "reused": True}
    else:
        counts = D.build_store(con, verbose=verbose)
        borrow, binfo = build_borrow(con, start=start, verbose=verbose)
    prices = con.execute("select symbol, date, close, adjclose from prices order by symbol, date").df(); adj, close = wide_prices(prices)
    effr = D.load_effr(); renames = con.execute("select * from symbol_history").df()
    if quick:
        borrow = borrow[borrow["date"] >= "2024-01-01"]
    checks = fee_checks(borrow, adj)
    out = {"as_of": str(borrow["date"].max().date()), "quick": quick, "universe": int(borrow["symbol"].nunique()), "store": counts, "borrow_model": {k: v for k, v in binfo.items()}, "fee_checks": checks, "strategies": {}}
    nav_series = {}
    strat_results = {}
    for name in names:
        r = run_strategy(name, borrow, adj, close, effr, books=("frictionless", "naive", "risk_only", "borrow_aware"))
        if "error" in r:
            out["strategies"][name] = r; continue
        out["strategies"][name] = r["summary"]; strat_results[name] = r
        for bk, res in r["results"].items():
            nav_series[f"{name}_{bk}"] = res.daily["nav"]
        if verbose:
            tbl = pd.DataFrame(r["summary"]["drag_table"]).T
            print(name, "\n", (tbl[["gross_price_pnl", "borrow_fees", "recall_impact", "opportunity_cost", "rebate", "net_return", "fee_wavg", "locate_failure_rate", "recalls_per_month"]] * 1).round(4).to_string())
    pd.DataFrame(nav_series).to_csv(os.path.join(RESULTS, "nav_daily.csv"))
    if HEADLINE in strat_results:
        hr = strat_results[HEADLINE]
        out["headline"] = {"strategy": HEADLINE, **hr["summary"]["recovery"], "nav0": hr["results"]["naive"].config.nav0, "gross_exposure": "100 % long / 100 % short of NAV, monthly rebalance"}
        naive_res = hr["results"]["naive"]
        naive_res.daily.to_csv(os.path.join(RESULTS, "headline_naive_daily.csv")); hr["results"]["borrow_aware"].daily.to_csv(os.path.join(RESULTS, "headline_aware_daily.csv"))
        naive_res.rebalances.to_csv(os.path.join(RESULTS, "headline_rebalances.csv"), index=False); naive_res.events.to_csv(os.path.join(RESULTS, "headline_events.csv"), index=False)
        reb = naive_res.rebalances.copy(); reb["year"] = pd.to_datetime(reb["date"]).dt.year; ev = naive_res.events; dd = naive_res.daily
        out["headline"]["locate_failure_by_year"] = {int(k): float(v) for k, v in (reb.groupby("year")["n_locate_failed"].sum() / reb.groupby("year")["n_short_requested"].sum()).items()}
        out["headline"]["most_failed_locates"] = {k: int(v) for k, v in ev[ev["event"] == "locate_failed"]["symbol"].value_counts().head(8).items()} if len(ev) else {}
        out["headline"]["opportunity_by_year"] = {int(k): float(v) for k, v in ((dd["shadow_pnl_short"] - dd["pnl_short"]) / dd["nav_open"]).groupby(dd.index.year).sum().items()}
        out["headline"]["fees_by_year"] = {int(k): float(v) for k, v in (dd["fees"] / dd["nav_open"]).groupby(dd.index.year).sum().items()}
        out["headline"]["nav_by_year"] = {int(k): float(v) for k, v in dd["nav"].groupby(dd.index.year).last().items()}
        out["financing"] = run_financing(naive_res, borrow, effr, con, renames)
        if verbose:
            print("financing", json.dumps({k: v for k, v in out["financing"].items() if k != "reconciliation"}, default=str)[:600]); print("recon", out["financing"]["reconciliation"]["breaks_by_type"], out["financing"]["reconciliation"]["overall_detection"])
        if not quick:
            out["sensitivity"] = sensitivity(HEADLINE, borrow, adj, close, effr)
        # the optimiser's two knobs: the alpha assumption and the tracking penalty
        out["alpha_sensitivity"] = []
        for sp, lam in (((0.005, LAM), (0.02, LAM), (0.01, 5.0), (0.01, 80.0)) if not quick else ((0.02, LAM),)):
            r2 = run_strategy(HEADLINE, borrow, adj, close, effr, lam=lam, alpha_spread=sp, books=("naive", "risk_only", "borrow_aware"), record_positions=False)
            out["alpha_sensitivity"].append({"alpha_spread_monthly": sp, "lam": lam, "avg_n_short": r2["summary"]["books"]["borrow_aware"]["avg_n_short"], **r2["summary"]["recovery"]})
    # monthly borrow statistics for the figures
    pd.DataFrame(checks["monthly"]).T.to_csv(os.path.join(RESULTS, "borrow_monthly.csv"))
    last = borrow[borrow["date"] == borrow["date"].max()][["symbol", "fee", "fee_floor_binds", "sir", "days_to_cover", "on_threshold", "utilisation", "p_locate", "recall_hazard", "score", "cap"]]
    last.sort_values("fee", ascending=False).to_csv(os.path.join(RESULTS, "borrow_last_day.csv"), index=False)
    out["seconds"] = round(time.time() - t0, 1); con.close()
    to_json(out, os.path.join(RESULTS, "run.json"))
    if verbose:
        print("done in", out["seconds"], "s")
    return out
