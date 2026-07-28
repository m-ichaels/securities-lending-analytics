"""Borrow-aware portfolio construction.  Given the signal scores on a rebalance date, the candidate longs (the top decile)
and shorts (the bottom two deciles, so that the optimiser has substitutes), each name's volatility, annual borrow fee,
locate probability and recall hazard, choose weights that minimise

    - sum_i alpha_i w_i  +  lambda sum_i sigma_i^2 (w_i - b_i)^2  +  sum_{i short} c_i |w_i|

with the long leg summing to +1, the short leg to -1 and 0 <= |w_i| <= w_max, where b is the equal-weighted decile
book (the naive book), alpha_i is the expected monthly return implied by the signal rank (a linear ramp from -spread/2
at the bottom of the candidate set to +spread/2 at the top) and c_i is the expected monthly cost of holding the short:
fee / 12, plus the probability of a recall within the month times the cost of a forced cover and re-entry, plus the
probability the locate fails times the alpha foregone.  The quadratic term is the tracking variance against the naive
book, so the optimiser only leaves the decile when the cost or the alpha says so: a short's weight moves by about
-c_i / (2 lambda sigma_i^2) and the freed weight goes to the cheapest near-substitutes.  The risk-only book is the same
problem with c = 0 (it differs from the naive book only through the alpha ramp), so the difference between the two
isolates the borrow term.  SLSQP on a few hundred variables takes well under a second."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RECALL_COST_BP = 20.0        # forced cover plus re-entry: two crossings at 10 bp each
ALPHA_MONTHLY_SPREAD = 0.010  # expected monthly return gap between the top and bottom of the candidate set (calibrated on the gross decile spread)
DAYS_PER_MONTH = 21


def expected_alpha(scores: pd.Series, spread: float = ALPHA_MONTHLY_SPREAD) -> pd.Series:
    r = scores.rank(pct=True)
    return (r - 0.5) * spread        # -spread/2 at the bottom, +spread/2 at the top


def borrow_cost_monthly(fee: pd.Series, hazard: pd.Series, p_locate: pd.Series, alpha_short: pd.Series) -> pd.Series:
    """expected monthly cost of holding a short in each name, per unit of notional"""
    recall_month = 1 - (1 - hazard) ** DAYS_PER_MONTH
    return fee / 12.0 + recall_month * RECALL_COST_BP * 1e-4 + (1 - p_locate) * alpha_short.abs()


def optimise(scores: pd.Series, borrow_day: pd.DataFrame, candidates: set[str], *, lam: float = 20.0, w_max: float = 0.04, n_decile: int = 10, borrow_aware: bool = True, alpha_spread: float = ALPHA_MONTHLY_SPREAD) -> tuple[pd.Series, pd.Series, dict]:
    s = scores[scores.index.isin(candidates)].dropna().sort_values()
    if len(s) < 40:
        return pd.Series(dtype=float), pd.Series(dtype=float), {}
    k = max(5, len(s) // n_decile); w_max = max(w_max, 1.5 / k)
    long_c = list(s.index[-k:]); short_c = list(s.index[:2 * k])            # the worst two deciles are short candidates
    b = borrow_day.set_index("symbol")
    alpha = expected_alpha(s, alpha_spread)
    names = long_c + short_c; n_l = len(long_c); n_s = len(short_c)
    a = alpha.reindex(names).fillna(0).values
    sig2 = (b["vol_3m"].reindex(names).fillna(0.4).clip(0.1, 2.0).values / np.sqrt(12)) ** 2       # monthly variance
    cost = borrow_cost_monthly(b["fee"].reindex(names).fillna(0.003), b["recall_hazard"].reindex(names).fillna(0.001), b["p_locate"].reindex(names).fillna(1.0), alpha.reindex(names).fillna(0)).to_numpy(copy=True)
    cost[:n_l] = 0.0
    if not borrow_aware:
        cost[:] = 0.0
    bench = np.concatenate([np.full(n_l, 1.0 / n_l), np.full(k, 1.0 / k), np.zeros(n_s - k)])     # the naive book
    sgn = np.concatenate([-np.ones(n_l), np.ones(n_s)])                                            # return = -sgn . a x
    # variables: long weights then short magnitudes, all >= 0
    def obj(x):
        return sgn @ (a * x) + lam * (sig2 @ (x - bench) ** 2) + cost @ x
    def grad(x):
        return sgn * a + 2 * lam * sig2 * (x - bench) + cost
    cons = [{"type": "eq", "fun": lambda x: x[:n_l].sum() - 1.0, "jac": lambda x: np.concatenate([np.ones(n_l), np.zeros(n_s)])},
            {"type": "eq", "fun": lambda x: x[n_l:].sum() - 1.0, "jac": lambda x: np.concatenate([np.zeros(n_l), np.ones(n_s)])}]
    res = minimize(obj, bench, jac=grad, bounds=[(0.0, w_max)] * len(names), constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    x = res.x if res.success else bench
    longs = pd.Series(x[:n_l], index=long_c); shorts = pd.Series(-x[n_l:], index=short_c)
    longs = longs[longs > 1e-6]; shorts = shorts[shorts < -1e-6]
    fee_s = b["fee"].reindex(shorts.index).fillna(0.003).values
    info = {"success": bool(res.success), "n_long": int(len(longs)), "n_short": int(len(shorts)), "expected_alpha": float(-(sgn @ (a * x))), "expected_borrow_cost": float(cost @ x), "expected_borrow_cost_naive": float(cost @ bench),
            "short_fee_weighted": float((fee_s * (-shorts.values)).sum()), "short_in_bottom_decile": float(-shorts[shorts.index.isin(s.index[:k])].sum()), "active_share": float(0.5 * np.abs(x - bench).sum())}
    return longs, shorts, info


def optimised_targets(naive: dict, borrow: pd.DataFrame, *, borrow_aware: bool = True, lam: float = 20.0, w_max: float = 0.04, alpha_spread: float = ALPHA_MONTHLY_SPREAD) -> dict:
    """re-solve every rebalance in a naive target dict (which carries the scores and the candidate set)"""
    out = {}
    for d, t in naive.items():
        day = borrow[borrow["date"] == d]
        lo, sh, info = optimise(t["scores"], day, t["candidates"], lam=lam, w_max=w_max, borrow_aware=borrow_aware, alpha_spread=alpha_spread)
        if lo.empty:
            continue
        out[d] = {"long": lo, "short": sh, "scores": t["scores"], "candidates": t["candidates"], "info": info}
    return out
