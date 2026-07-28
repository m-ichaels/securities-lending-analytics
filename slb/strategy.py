"""Three systematic long-short signals on the universe, monthly: 12-1 momentum, published short interest (Boehmer,
Huszar and Jordan: high short interest predicts low returns, so the strategy is long the least-shorted decile and short
the most-shorted one), and one-month reversal.  Deciles are equal-weighted, 100 % long and 100 % short of NAV.  The
candidate set on each rebalance date is the names with a price above $5 and dollar volume above $5m."""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = {"MOM": "12-1 momentum: long the top decile of 12-month return skipping the last month, short the bottom",
           "SIR": "short interest: long the least-shorted decile (short interest / shares outstanding, as published), short the most-shorted",
           "REV": "one-month reversal: long the worst decile of the last month, short the best"}


def month_ends(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(dates, index=dates); return [pd.Timestamp(x) for x in s.groupby([dates.year, dates.month]).max().values]


def candidate_set(panel_day: pd.DataFrame, min_price: float = 5.0, min_dollar_adv: float = 5e6) -> pd.DataFrame:
    return panel_day[(panel_day["close"] >= min_price) & (panel_day["dollar_adv"] >= min_dollar_adv)]


def signal_scores(adj: pd.DataFrame, borrow: pd.DataFrame, d: pd.Timestamp, name: str) -> pd.Series:
    """higher score = more attractive to hold long (and its negative more attractive to short)"""
    hist = adj.loc[:d]
    if name == "MOM":
        if len(hist) < 260:
            return pd.Series(dtype=float)
        return (hist.iloc[-22] / hist.iloc[-253] - 1.0).dropna()
    if name == "REV":
        if len(hist) < 30:
            return pd.Series(dtype=float)
        return -(hist.iloc[-1] / hist.iloc[-22] - 1.0).dropna()
    if name == "SIR":
        day = borrow[borrow["date"] == d].set_index("symbol")["sir"].dropna()
        return -day
    raise ValueError(name)


def decile_portfolio(scores: pd.Series, candidates: set[str], n_decile: int = 10) -> tuple[pd.Series, pd.Series]:
    s = scores[scores.index.isin(candidates)].dropna().sort_values()
    if len(s) < 40:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    k = max(5, len(s) // n_decile)
    longs = pd.Series(1.0 / k, index=s.index[-k:]); shorts = pd.Series(-1.0 / k, index=s.index[:k])
    return longs, shorts


def naive_targets(adj: pd.DataFrame, borrow: pd.DataFrame, rebalance_dates: list, name: str) -> dict:
    """{date: {'long': Series, 'short': Series, 'scores': Series, 'candidates': set}}"""
    out = {}
    for d in rebalance_dates:
        day = borrow[borrow["date"] == d]
        if day.empty:
            continue
        cands = set(candidate_set(day)["symbol"]); sc = signal_scores(adj, borrow, d, name)
        if sc.empty:
            continue
        lo, sh = decile_portfolio(sc, cands)
        if lo.empty:
            continue
        out[d] = {"long": lo, "short": sh, "scores": sc[sc.index.isin(cands)], "candidates": cands}
    return out
