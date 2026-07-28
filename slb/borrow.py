"""The borrow-cost model.  Daily panel of what a desk can observe about each name (short interest ratio and days to cover
from FINRA with its publication lag, Regulation SHO threshold membership, fails to deliver, price and dollar volume), a
specialness score, and the mapping from the score to an indicative annual borrow fee.  When Interactive Brokers
snapshots exist in the store the mapping is fitted to them (isotonic regression of the IB fee on the score); otherwise it
is the quantile map fixed in advance from the published cross-section of fees (D'Avolio 2002; Engelberg, Reed and
Ringgenberg 2018; Muravyev, Pearson and Pollet 2022).  Lendable supply, utilisation, the locate probability and the daily
recall hazard follow from the same panel with stated assumptions."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# target cross-section of annual fees for exchange-listed common stocks (percentile -> fee), fixed before the data
FEE_QUANTILES = [(0.00, 0.0025), (0.50, 0.0030), (0.75, 0.0040), (0.90, 0.0100), (0.95, 0.0250), (0.98, 0.0700), (0.99, 0.1500), (0.995, 0.3000), (1.00, 1.0000)]
GC_FEE = 0.0025
PUBLICATION_LAG_BD = 9         # FINRA publishes about nine business days after the settlement date
FTD_LAG_DAYS = 30              # the SEC posts each half-month of fails about a month later
LENDABLE_SHARE = {"large": 0.30, "mid": 0.25, "small": 0.18}   # lendable supply as a share of shares outstanding
RECALL_H0, RECALL_H1, RECALL_H2 = 0.0007, 8.0, 2.0          # daily hazard = h0 (1 + h1 u^2) (1 + h2 threshold)
UTIL_FLOOR = [(0.0, 0.0025), (0.7, 0.0025), (0.85, 0.0075), (1.0, 0.0250), (1.2, 0.0750), (1.5, 0.2000)]     # utilisation -> minimum fee


def zscore(x: pd.Series, clip: float = 3.0) -> pd.Series:
    s = x.std(); m = x.mean()
    if not np.isfinite(s) or s == 0:
        return pd.Series(0.0, index=x.index)
    return ((x - m) / s).clip(-clip, clip)


def fee_from_percentile(p: np.ndarray) -> np.ndarray:
    xs = np.array([q for q, _ in FEE_QUANTILES]); ys = np.log(np.array([f for _, f in FEE_QUANTILES]))
    return np.exp(np.interp(p, xs, ys))


def cap_bucket(mc: float) -> str:
    return "large" if mc >= 1e10 else "mid" if mc >= 2e9 else "small"


def business_days_after(d: pd.Timestamp, n: int, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    i = trading_days.searchsorted(d); j = min(i + n, len(trading_days) - 1); return trading_days[j]


def daily_panel(prices: pd.DataFrame, si: pd.DataFrame, th: pd.DataFrame, ftd: pd.DataFrame, securities: pd.DataFrame, events: pd.DataFrame | None = None, start: str = "2019-06-01") -> pd.DataFrame:
    """One row per (date, symbol) with the observable features as known on that date."""
    px = prices[prices["date"] >= start].copy(); px["date"] = pd.to_datetime(px["date"])
    days = pd.DatetimeIndex(sorted(px["date"].unique()))
    sec = securities.set_index("symbol")
    # split-adjusted shares outstanding back in time
    splits = {}
    if events is not None:
        for r in events[events["type"] == "split"].itertuples():
            splits.setdefault(r.symbol, []).append((pd.Timestamp(r.ex_date), float(r.value)))
    # dollar volume and returns per symbol
    px = px.sort_values(["symbol", "date"])
    px["adv_shares"] = px.groupby("symbol")["volume"].transform(lambda v: v.rolling(20, min_periods=5).mean())
    px["dollar_adv"] = px["adv_shares"] * px["close"]
    px["ret_1m"] = px.groupby("symbol")["adjclose"].transform(lambda s: s / s.shift(21) - 1)
    px["vol_3m"] = px.groupby("symbol")["adjclose"].transform(lambda s: np.log(s).diff().rolling(63, min_periods=20).std() * np.sqrt(252))
    # short interest as published: settlement date + lag, forward-filled to daily
    si = si.copy(); si["pub_date"] = [business_days_after(d, PUBLICATION_LAG_BD, days) for d in si["settle_date"]]
    si = si.sort_values(["symbol", "pub_date"])
    frames = []
    for sym, g in px.groupby("symbol"):
        g = g.set_index("date")
        so = float(sec.loc[sym, "shares_outstanding"]) if sym in sec.index and pd.notna(sec.loc[sym, "shares_outstanding"]) else np.nan
        if np.isfinite(so):
            adj = pd.Series(so, index=g.index)
            for ex, ratio in splits.get(sym, []):
                adj[g.index < ex] = adj[g.index < ex] / ratio
            g["shares_out"] = adj
        else:
            g["shares_out"] = np.nan
        s = si[si["symbol"] == sym].copy()
        if len(s):
            # the ratio at the settlement date is what carries forward: a reverse split between the settlement and today
            # (common in the small caps with the largest short interest) would otherwise leave a pre-split count over a
            # post-split share count
            so_at_settle = g["shares_out"].reindex(s["settle_date"], method="ffill").values
            s["sir_settle"] = s["shares_short"].values / so_at_settle
            s = s.set_index("pub_date")[["sir_settle", "days_to_cover", "adv"]]
            s = s[~s.index.duplicated(keep="last")].reindex(g.index, method="ffill")
            g = g.join(s)
        else:
            g["sir_settle"] = np.nan; g["days_to_cover"] = np.nan; g["adv"] = np.nan
        frames.append(g.reset_index())
    panel = pd.concat(frames, ignore_index=True)
    panel["sir"] = panel["sir_settle"].clip(0, 1.5)
    panel["shares_short"] = panel["sir"] * panel["shares_out"]          # in today's share count
    # threshold membership, published the next morning
    if len(th):
        t = th.copy(); t["date"] = pd.to_datetime(t["date"]) + pd.Timedelta(days=1); t["on_threshold"] = True
        t = t.groupby(["symbol", "date"]).size().rename("n").reset_index(); t["on_threshold"] = True
        panel = panel.merge(t[["symbol", "date", "on_threshold"]], on=["symbol", "date"], how="left")
        # a listing lasts until the next list without the name; treat membership as valid for five business days after each listing
        panel["on_threshold"] = panel.groupby("symbol")["on_threshold"].transform(lambda s: s.fillna(False).astype(float).rolling(5, min_periods=1).max().astype(bool))
    else:
        panel["on_threshold"] = False
    # fails to deliver relative to volume, as published a month later
    if len(ftd):
        f = ftd.copy(); f["pub_date"] = pd.to_datetime(f["settle_date"]) + pd.Timedelta(days=FTD_LAG_DAYS)
        f = f.groupby(["symbol", "pub_date"])["quantity"].mean().rename("ftd_qty").reset_index().sort_values(["symbol", "pub_date"])
        parts = []
        for sym, g in panel.groupby("symbol"):
            ff = f[f["symbol"] == sym].set_index("pub_date")["ftd_qty"]; ff = ff[~ff.index.duplicated(keep="last")]
            g = g.set_index("date"); g["ftd_qty"] = ff.reindex(g.index, method="ffill").values; parts.append(g.reset_index())
        panel = pd.concat(parts, ignore_index=True)
        panel["ftd_ratio"] = (panel["ftd_qty"].fillna(0) / panel["adv_shares"].fillna(panel["volume"]).clip(lower=1)).clip(0, 5)
    else:
        panel["ftd_ratio"] = 0.0
    panel["cap"] = [cap_bucket(sec.loc[s, "market_cap"]) if s in sec.index and pd.notna(sec.loc[s, "market_cap"]) else "mid" for s in panel["symbol"]]
    return panel


def specialness_score(panel: pd.DataFrame) -> pd.Series:
    """cross-sectional score per day: z(short interest ratio) + 0.5 z(days to cover) + 1.5 threshold + 0.5 z(fails ratio)
    + 0.5 z(-log dollar volume) + 0.3 z(-log price)"""
    def per_day(g):
        z = zscore(np.log1p(g["sir"].fillna(0) * 100)) + 0.5 * zscore(np.log1p(g["days_to_cover"].fillna(0))) + 1.5 * g["on_threshold"].astype(float) + 0.5 * zscore(np.log1p(g["ftd_ratio"].fillna(0))) + 0.5 * zscore(-np.log(g["dollar_adv"].fillna(1e6).clip(lower=1e4))) + 0.3 * zscore(-np.log(g["close"].clip(lower=0.1)))
        return z
    return panel.groupby("date", group_keys=False).apply(per_day)


def pav(y: np.ndarray) -> np.ndarray:
    """pool-adjacent-violators: the non-decreasing least-squares fit to y (already ordered by the regressor)"""
    merged: list[list[float]] = []       # [mean, weight, first index, last index]
    for k, v in enumerate(y):
        merged.append([float(v), 1.0, k, k])
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            a, c = merged.pop(), merged.pop(); merged.append([(a[0] * a[1] + c[0] * c[1]) / (a[1] + c[1]), a[1] + c[1], c[2], a[3]])
    fit = np.zeros(len(y))
    for b in merged:
        fit[int(b[2]):int(b[3]) + 1] = b[0]
    return fit


def fee_from_score(panel: pd.DataFrame, ib: pd.DataFrame | None = None) -> tuple[pd.Series, dict]:
    """Map the score to a fee.  With IB snapshots: isotonic regression of the IB fee on the score over the overlapping days,
    then applied to every day.  Without: the daily percentile of the score through the literature quantile map."""
    info = {"source": "quantile_map", "quantiles": FEE_QUANTILES}
    if ib is not None and len(ib):
        m = panel.merge(ib.rename(columns={"fee_rate": "ib_fee"})[["date", "symbol", "ib_fee"]], on=["date", "symbol"], how="inner").dropna(subset=["ib_fee", "score"])
        if len(m) >= 200:
            from scipy.stats import spearmanr
            order = np.argsort(m["score"].values); x = m["score"].values[order]; y = m["ib_fee"].values[order] / 100.0
            fit = pav(y); rho = spearmanr(m["score"], m["ib_fee"]).correlation
            info = {"source": "ib_isotonic", "n_pairs": int(len(m)), "days": int(m["date"].nunique()), "spearman": float(rho), "median_ib_fee": float(np.median(y))}
            fee = np.interp(panel["score"].values, x, fit)
            return pd.Series(np.clip(fee, GC_FEE, 2.0), index=panel.index), info
    pct = panel.groupby("date")["score"].rank(pct=True).values
    return pd.Series(fee_from_percentile(pct), index=panel.index), info


def fee_floor(u: np.ndarray) -> np.ndarray:
    """the fee a name cannot be below at a given utilisation of its lendable supply: general collateral until about 70 %
    of the supply is out, then steeply higher (Kolasinski, Reed and Ringgenberg 2013: fees are insensitive to demand until
    supply is nearly exhausted).  Log-linear between the knots."""
    xs = np.array([k for k, _ in UTIL_FLOOR]); ys = np.log(np.array([f for _, f in UTIL_FLOOR]))
    return np.exp(np.interp(u, xs, ys))


def supply_and_hazard(panel: pd.DataFrame) -> pd.DataFrame:
    """lendable supply, utilisation, the utilisation floor on the fee, the locate probability and the daily recall hazard
    from the panel and the stated assumptions"""
    out = panel.copy()
    out["lendable"] = out["shares_out"] * out["cap"].map(LENDABLE_SHARE)
    out["utilisation"] = (out["shares_short"] / out["lendable"]).clip(0, 1.5).fillna(0.3)
    u = out["utilisation"].values; thr = out["on_threshold"].astype(float).values
    if "fee" in out:
        floor = fee_floor(u); out["fee_floor_binds"] = floor > out["fee"].values + 1e-12; out["fee"] = np.maximum(out["fee"].values, floor)
    p = np.where(u <= 0.5, 1.0, np.where(u <= 1.0, 1.0 - 1.4 * (u - 0.5), 0.3 - 0.4 * np.clip(u - 1.0, 0, 0.5)))
    out["p_locate"] = np.clip(p * (1 - 0.5 * thr), 0.05, 1.0)
    out["recall_hazard"] = RECALL_H0 * (1 + RECALL_H1 * np.minimum(u, 1.2) ** 2) * (1 + RECALL_H2 * thr)
    return out


def build_borrow_table(panel: pd.DataFrame, ib: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    panel = panel.copy(); panel["score"] = specialness_score(panel)
    fee, info = fee_from_score(panel, ib); panel["fee"] = fee.values; panel["source"] = info["source"]
    out = supply_and_hazard(panel)
    cols = ["date", "symbol", "sir", "days_to_cover", "on_threshold", "ftd_ratio", "score", "fee", "fee_floor_binds", "lendable", "utilisation", "p_locate", "recall_hazard", "source"]
    info["fee_floor_binds_share"] = float(out["fee_floor_binds"].mean())
    return out[cols + ["close", "adjclose", "volume", "dollar_adv", "ret_1m", "vol_3m", "shares_out", "shares_short", "cap"]], info


def fee_distribution(borrow: pd.DataFrame) -> dict:
    last = borrow[borrow["date"] == borrow["date"].max()]
    q = last["fee"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    return {"date": str(last["date"].iloc[0].date()), "n": int(len(last)), "share_gc_below_1pct": float((last["fee"] < 0.01).mean()), "share_special_above_5pct": float((last["fee"] > 0.05).mean()), "mean_fee": float(last["fee"].mean()), "quantiles": {f"p{int(100 * k)}": float(v) for k, v in q.items()},
            "on_threshold": int(last["on_threshold"].sum()), "mean_sir": float(last["sir"].mean()), "utilisation_above_0_8": int((last["utilisation"] > 0.8).sum()), "fee_floor_binds": int(last["fee_floor_binds"].sum()) if "fee_floor_binds" in last else 0}
