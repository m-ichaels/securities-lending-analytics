#!/usr/bin/env python3
"""Figures from results/*.json and results/*.csv.   python scripts/plots.py [results] [results/figures]"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
F = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "figures")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 130})
BOOKS = {"frictionless": ("no borrow frictions", "C7"), "naive": ("naive decile book", "C3"), "risk_only": ("optimiser, no borrow term", "C1"), "borrow_aware": ("borrow-aware optimiser", "C2")}


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def csv(name, **kw):
    p = os.path.join(R, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(F, name)); plt.close(fig); print("wrote", os.path.join(F, name))


run = load("run.json")


def fig_fee():
    last = csv("borrow_last_day.csv"); monthly = csv("borrow_monthly.csv", index_col=0)
    if last is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    fee = last["fee"] * 100
    bins = np.logspace(np.log10(0.2), np.log10(100), 40)
    ax[0].hist(fee, bins=bins, color="C0", alpha=0.8); ax[0].set_xscale("log"); ax[0].set_xlabel("modelled annual borrow fee (%)"); ax[0].set_ylabel("names")
    ax[0].set_title(f"fee cross-section, {last['fee'].notna().sum()} names, {run['as_of'] if run else ''}")
    for q, lab in ((0.5, "median"), (0.9, "p90"), (0.99, "p99")):
        v = fee.quantile(q); ax[0].axvline(v, color="C3", lw=0.8, ls="--"); ax[0].text(v, ax[0].get_ylim()[1] * 0.9, f"{lab} {v:.1f}%", rotation=90, va="top", fontsize=6, color="C3")
    thr = last["on_threshold"].astype(bool)
    ax[1].scatter(last.loc[~thr, "sir"] * 100, last.loc[~thr, "fee"] * 100, s=6, alpha=0.5, label="not on threshold list")
    ax[1].scatter(last.loc[thr, "sir"] * 100, last.loc[thr, "fee"] * 100, s=14, color="C3", label="on Reg SHO threshold list")
    ax[1].set_yscale("log"); ax[1].set_xlabel("short interest / shares outstanding (%)"); ax[1].set_ylabel("fee (%)"); ax[1].legend(); ax[1].set_title("fee against published short interest")
    if monthly is not None:
        m = monthly.copy(); m.index = pd.to_datetime(m.index)
        ax[2].plot(m.index, m["share_special"] * 100, color="C3", label="share of names with fee > 5 %")
        ax[2].plot(m.index, m["mean_util"] * 100, color="C0", label="mean utilisation of lendable supply")
        ax[2].plot(m.index, m["mean_sir"] * 100, color="C2", label="mean short interest ratio")
        ax2 = ax[2].twinx(); ax2.plot(m.index, m["on_threshold"] / 21, color="C1", lw=0.8, label="names on threshold list (avg per day)"); ax2.set_ylabel("threshold names / day")
        ax[2].set_ylabel("%"); ax[2].set_title("the universe through time"); h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, loc="upper left")
    save(fig, "fee_cross_section.png")


def fig_waterfall():
    if not run or "headline" not in run:
        return
    name = run["headline"]["strategy"]; st = run["strategies"][name]["drag_table"]
    comps = [("borrow fees", "borrow_fees", -1), ("recalls: forced covers" + chr(10) + "and re-entries", "recall_impact", -1), ("missed shorts" + chr(10) + "(shadow leg minus real)", "opportunity_cost", -1), ("rebalance impact", "rebalance_impact", -1), ("short rebate + cash", None, 1)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(comps)); w = 0.38
    for j, bk in enumerate(("naive", "borrow_aware")):
        t = st[bk]; vals = []
        for lab, key, sgn in comps:
            v = (t["rebate"] + t["cash_financing"]) if key is None else sgn * t[key]
            vals.append(v * 100)
        bars = ax[0].bar(x + (j - 0.5) * w, vals, w, color=BOOKS[bk][1], label=f"{BOOKS[bk][0]}: gross {100 * t['gross_price_pnl']:.1f} %, net {100 * t['net_return']:.1f} %")
        for b, v in zip(bars, vals):
            ax[0].text(b.get_x() + b.get_width() / 2, v + (0.06 if v >= 0 else -0.16), f"{v:+.2f}", ha="center", fontsize=6.5)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xticks(x); ax[0].set_xticklabels([c[0] for c in comps], fontsize=7); ax[0].set_ylabel("% of NAV per year (annualised)"); ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_title(f"{name} $500m 100/100: what sits between gross price P&L and net")
    labels = ["fees + recalls" + chr(10) + "(certain)", "+ missed shorts" + chr(10) + "(all-in)"]
    for j, bk in enumerate(("naive", "risk_only", "borrow_aware")):
        t = st[bk]; vals = [100 * t["drag_certain"], 100 * t["drag_borrow"]]
        bars = ax[1].bar(np.arange(2) + (j - 1) * 0.27, vals, 0.27, color=BOOKS[bk][1], label=BOOKS[bk][0])
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.12), f"{v:.2f}", ha="center", fontsize=6.5)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(np.arange(2)); ax[1].set_xticklabels(labels, fontsize=7); ax[1].legend(fontsize=7); ax[1].set_title("the drag from borrow and recalls, % of NAV / yr")
    save(fig, "drag_waterfall.png")


def fig_nav():
    nav = csv("nav_daily.csv", index_col=0, parse_dates=True); d = csv("headline_naive_daily.csv", index_col=0, parse_dates=True)
    if nav is None or not run or "headline" not in run:
        return
    name = run["headline"]["strategy"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for bk, (lab, col) in BOOKS.items():
        c = f"{name}_{bk}"
        if c in nav:
            ax[0].plot(nav.index, nav[c] / nav[c].iloc[0], color=col, lw=1, label=lab)
    ax[0].set_title(f"{name}: NAV, $500m book, 100/100 (log scale)"); ax[0].legend(); ax[0].set_ylabel("NAV / NAV0"); ax[0].set_yscale("log")
    if d is not None:
        fees = (d["fees"] / d["nav_open"]).cumsum() * 100; both = ((d["fees"] + d["recall_impact"]) / d["nav_open"]).cumsum() * 100
        ax[1].plot(d.index, fees, color="C3", label="borrow fees")
        ax[1].plot(d.index, both, color="C1", label="+ recall covers and re-entries")
        a = csv("headline_aware_daily.csv", index_col=0, parse_dates=True)
        if a is not None:
            ax[1].plot(a.index, ((a["fees"] + a["recall_impact"]) / a["nav_open"]).cumsum() * 100, color="C2", label="same, borrow-aware book")
        ax[1].set_ylabel("cumulative return given up, % of NAV"); ax[1].set_title("what the naive book paid"); ax[1].legend()
        ax[2].plot(d.index, d["fee_wavg"] * 100, color="C3", lw=0.8, label="weighted fee on the short leg (%)")
        ax2 = ax[2].twinx(); ax2.plot(d.index, d["n_missing"], color="C0", lw=0.6, label="intended shorts not held"); ax2.set_ylabel("names")
        ax[2].set_ylabel("%"); ax[2].set_title("short leg: fee and unavailability"); h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2)
    save(fig, "nav.png")


def fig_strategies():
    if not run:
        return
    names = [n for n in run["strategies"] if "drag_table" in run["strategies"][n]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    comps = [("borrow_fees", "fees", "C3"), ("recall_impact", "recalls (covers + re-entries)", "C1")]
    x = np.arange(len(names)); w = 0.38
    for j, bk in enumerate(("naive", "borrow_aware")):
        bottom = np.zeros(len(names))
        for key, lab, col in comps:
            v = np.array([run["strategies"][n]["drag_table"][bk][key] * 100 for n in names])
            ax[0].bar(x + (j - 0.5) * w, v, w, bottom=bottom, color=col, alpha=1.0 if bk == "naive" else 0.5, label=f"{lab}, {BOOKS[bk][0]}"); bottom += v
        for i, n in enumerate(names):
            ax[0].text(x[i] + (j - 0.5) * w, bottom[i] + 0.05, f"{bottom[i]:.2f}", ha="center", fontsize=6.5)
        opp = [run["strategies"][n]["drag_table"][bk]["opportunity_cost"] * 100 for n in names]
        ax[0].scatter(x + (j - 0.5) * w, opp, marker="D", s=22, color="C0", alpha=1.0 if bk == "naive" else 0.5, zorder=3, label=f"missed shorts (+ = cost), {BOOKS[bk][0]}")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_ylabel("% of NAV per year"); ax[0].set_title("fees + recalls (bars), missed shorts (diamonds); naive solid, borrow-aware light"); ax[0].legend(fontsize=6, ncol=2, loc="upper right"); ax[0].set_ylim(bottom=min(-2.0, ax[0].get_ylim()[0]))
    for j, bk in enumerate(("naive", "borrow_aware")):
        v = [run["strategies"][n]["books"][bk]["ann_return"] * 100 for n in names]
        ax[1].bar(x + (j - 0.5) * w, v, w, color=BOOKS[bk][1], label=BOOKS[bk][0])
    v = [run["strategies"][n]["books"]["frictionless"]["ann_return"] * 100 for n in names]
    ax[1].scatter(x, v, color="k", marker="_", s=300, label="no borrow frictions", zorder=3)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylabel("net return, % / yr"); ax[1].set_title("net return by book"); ax[1].legend(); ax[1].axhline(0, color="k", lw=0.5)
    save(fig, "strategies.png")


def fig_sensitivity():
    if not run or "sensitivity" not in run:
        return
    g = pd.DataFrame(run["sensitivity"]["grid"]); s = pd.DataFrame(run["sensitivity"]["seeds"])
    cn, ca = ("certain_naive", "certain_aware") if "certain_naive" in g else ("drag_naive", "drag_borrow_aware")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    fs = g[g["hazard_scale"] == 1.0].sort_values("fee_scale")
    ax[0].plot(fs["fee_scale"], fs[cn] * 100, "o-", color="C3", label="fees + recalls, naive"); ax[0].plot(fs["fee_scale"], fs[ca] * 100, "o-", color="C2", label="fees + recalls, borrow-aware")
    ax[0].plot(fs["fee_scale"], (fs[cn] - fs[ca]) * 100, "s--", color="C1", label="recovery in fees + recalls"); ax[0].plot(fs["fee_scale"], fs["gain"] * 100, "^:", color="C0", label="net gain of the optimiser (all-in)")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("fee scale (x the modelled fees)"); ax[0].set_ylabel("% of NAV / yr"); ax[0].legend(); ax[0].set_title("fee level")
    hs = g[g["fee_scale"] == 1.0].sort_values("hazard_scale")
    ax[1].plot(hs["hazard_scale"], hs[cn] * 100, "o-", color="C3", label="fees + recalls, naive"); ax[1].plot(hs["hazard_scale"], hs[ca] * 100, "o-", color="C2", label="fees + recalls, borrow-aware")
    ax[1].plot(hs["hazard_scale"], hs["gain"] * 100, "^:", color="C0", label="net gain of the optimiser (all-in)")
    ax2 = ax[1].twinx(); ax2.plot(hs["hazard_scale"], hs["recalls_naive"], "x-", color="C7", lw=0.8, label="recalls per month, naive"); ax2.set_ylabel("recalls / month")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("recall hazard scale"); h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=6); ax[1].set_title("recall intensity")
    scn = "certain_naive" if "certain_naive" in s else "drag_naive"; sca = "certain_aware" if "certain_aware" in s else "drag_borrow_aware"
    ax[2].bar(s["seed"] - 0.2, s[scn] * 100, 0.4, color="C3", label="fees + recalls, naive"); ax[2].bar(s["seed"] + 0.2, s[sca] * 100, 0.4, color="C2", label="fees + recalls, borrow-aware")
    ax[2].scatter(s["seed"], s["opportunity_naive"] * 100, marker="D", color="C0", zorder=3, label="missed shorts, naive (+ = cost)")
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set_xlabel("random seed (locate and recall draws)"); ax[2].set_xticks(s["seed"]); ax[2].legend(fontsize=6); ax[2].set_title("draw-to-draw variation")
    save(fig, "sensitivity.png")


def fig_recon():
    if not run or "financing" not in run:
        return
    rc = run["financing"]["reconciliation"]; br = csv("fin_breaks.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    sd = rc["seeded"]; names = list(sd); n = [sd[k]["seeded"] for k in names]; hit = [sd[k]["detected_as_expected"] for k in names]
    y = np.arange(len(names)); ax[0].barh(y - 0.2, n, 0.4, color="C7", label="seeded"); ax[0].barh(y + 0.2, hit, 0.4, color="C2", label="found and classified")
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{k} -> {sd[k]['expected_type']}" for k in names], fontsize=6); ax[0].set_xscale("log"); ax[0].legend(); ax[0].set_title(f"seeded discrepancies: {100 * rc['overall_detection']:.1f} % found", fontsize=8)
    bt = rc["breaks_by_type"]; usd = rc["breaks_usd_by_type"]
    keys = sorted(bt, key=lambda k: -bt[k]); ax[1].bar(range(len(keys)), [bt[k] for k in keys], color="C0"); ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels(keys, rotation=30, ha="right", fontsize=6); ax[1].set_title(f"breaks: {rc['breaks']} of {rc['keys']:,} keys, {rc['false_positives']} on clean keys", fontsize=8); ax[1].set_yscale("log")
    ax[2].bar(range(len(keys)), [abs(usd.get(k, 0)) for k in keys], color=["C3" if usd.get(k, 0) < 0 else "C2" for k in keys]); ax[2].set_xticks(range(len(keys))); ax[2].set_xticklabels(keys, rotation=30, ha="right", fontsize=6); ax[2].set_yscale("log"); ax[2].set_title("$ of the breaks (red: PB charges more / pays less)", fontsize=8)
    save(fig, "recon.png")


def fig_fee_returns():
    if not run:
        return
    fc = run["fee_checks"]; fwd = fc["forward_return_by_fee_bucket"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    keys = [k for k in ("gc", "1-5%", ">5%") if k in fwd]
    m = [fwd[k]["mean_monthly_return"] * 100 for k in keys]; t = [fwd[k]["t_stat"] or 0 for k in keys]
    ax[0].bar(range(len(keys)), m, color=["C2", "C1", "C3"][:len(keys)])
    for i, (v, tt) in enumerate(zip(m, t)):
        ax[0].text(i, v + (0.05 if v >= 0 else -0.15), f"{v:+.2f} %/mo (t={tt:.1f})", ha="center", fontsize=7)
    ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(["general collateral (< 1 %)", "1-5 %", "special (> 5 %)"][:len(keys)]); ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("next-month return, equal weight (%)"); ax[0].set_title("forward return by modelled fee bucket")
    monthly = csv("borrow_monthly.csv", index_col=0)
    if monthly is not None:
        mm = monthly.copy(); mm.index = pd.to_datetime(mm.index); ax[1].plot(mm.index, mm["p90_fee"] * 100, color="C3", label="p90 fee"); ax[1].plot(mm.index, mm["mean_fee"] * 100, color="C0", label="mean fee"); ax[1].set_ylabel("%"); ax[1].legend(); ax[1].set_title("fee level through time (percentile map with the utilisation floor)")
    save(fig, "fee_returns.png")


def fig_availability():
    reb = csv("headline_rebalances.csv", parse_dates=["date"]); d = csv("headline_naive_daily.csv", index_col=0, parse_dates=True)
    if reb is None or d is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].bar(reb["date"], reb["n_locate_failed"], width=20, color="C3", label="locates failed"); ax[0].plot(reb["date"], reb["n_short_requested"], color="C7", lw=0.8, label="shorts requested"); ax[0].legend(); ax[0].set_title("locates at each monthly rebalance")
    rm = d["n_recalls"].resample("ME").sum(); ax[1].bar(rm.index, rm.values, width=20, color="C1", label="recalls per month"); ax[1].plot(d.index, d["n_short"], color="C7", lw=0.6, label="shorts held"); ax[1].legend(); ax[1].set_title("recalls")
    ax[2].plot(d.index, d["excess_equity_pm"] / d["nav"] * 100, color="C2", label="excess equity, portfolio margin"); ax[2].plot(d.index, d["excess_equity_regt"] / d["nav"] * 100, color="C0", label="excess equity, Reg T maintenance")
    ax[2].plot(d.index, (d["nav"] - d["margin_regt_initial"]) / d["nav"] * 100, color="C3", lw=0.8, label="vs Reg T initial (50/50, applies at trade time)"); ax[2].axhline(0, color="k", lw=0.5); ax[2].set_ylabel("% of NAV"); ax[2].legend(); ax[2].set_title("margin headroom")
    save(fig, "availability.png")


if __name__ == "__main__":
    for f in (fig_fee, fig_waterfall, fig_nav, fig_strategies, fig_sensitivity, fig_recon, fig_fee_returns, fig_availability):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("skipped", f.__name__, repr(e))
