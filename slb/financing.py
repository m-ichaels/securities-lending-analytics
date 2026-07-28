"""The financing layer.  Daily accruals on the book's own positions (borrow fee on the short notional and the rebate on the
short proceeds, both ACT/360; interest on the cash balance at EFFR plus a spread), the margin requirement under
Regulation T and portfolio margin, a simulated prime-broker statement of the same accruals with seeded discrepancies of
the kinds a desk actually meets (a stale fee rate, ACT/365 on a rebate, a missing line, a duplicated line, fees on 102 %
collateral, a stale mark, an old ticker after a rename, a buy-in charge, yesterday's funding rate), and the
reconciliation that finds and classifies them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KINDS = ("borrow_fee", "short_rebate", "cash_interest")


@dataclass
class FinConfig:
    day_count: int = 360
    financing_spread: float = 0.005
    cash_spread: float = -0.001
    pb_collateral: float = 1.02          # the PB accrues the fee on the cash collateral, 102 % of the mark
    tol_usd: float = 0.50
    seed: int = 7


# ---- margin ----------------------------------------------------------------------------------------------------------------
def margin_requirements(long_notional: float, short_notional: float, nav: float, short_positions: pd.DataFrame | None = None) -> dict:
    """Regulation T (initial 50 % each leg; maintenance 25 % long and 30 % short, with the FINRA 4210 per-share floor on the
    shorts when positions are given: 100 % or $2.50 a share below $5, $5 a share between $5 and $16.67) and portfolio
    margin (15 % of gross for a diversified equity book)."""
    regt_initial = 0.5 * long_notional + 0.5 * short_notional
    regt_maint = 0.25 * long_notional + 0.30 * short_notional
    if short_positions is not None and len(short_positions):
        q = short_positions["shares"].abs(); p = short_positions["close"]
        per_name = np.where(p < 5.0, np.maximum(q * 2.5, q * p), np.maximum(q * 5.0, 0.30 * q * p))
        regt_maint = 0.25 * long_notional + float(per_name.sum())
    pm = 0.15 * (long_notional + short_notional)
    return {"regt_initial": regt_initial, "regt_maintenance": regt_maint, "portfolio_margin": pm, "excess_regt_initial": nav - regt_initial, "excess_regt_maintenance": nav - regt_maint, "excess_pm": nav - pm,
            "max_gross_regt_initial": nav / 0.5, "max_gross_pm": nav / 0.15, "gross": long_notional + short_notional, "leverage": (long_notional + short_notional) / nav if nav else np.nan}


# ---- book accruals -----------------------------------------------------------------------------------------------------------
def book_accruals(positions: pd.DataFrame, daily: pd.DataFrame, fee_w: pd.DataFrame, effr: pd.Series, cfg: FinConfig | None = None, book: str = "LS") -> pd.DataFrame:
    """one line per (date, symbol, kind) from the backtest's positions: fee and rebate on each short, interest on the cash balance"""
    cfg = cfg or FinConfig()
    pos = positions.copy(); pos["date"] = pd.to_datetime(pos["date"]); pos["notional"] = pos["shares"].abs() * pos["close"]
    sh = pos[pos["shares"] < 0].copy()
    fl = fee_w.stack().rename("fee").reset_index(); fl.columns = ["date", "symbol", "fee"]
    sh = sh.merge(fl, on=["date", "symbol"], how="left"); sh["fee"] = sh["fee"].fillna(0.003)
    r = effr.reindex(sh["date"], method="ffill").values; sh["effr"] = r
    fee = pd.DataFrame({"date": sh["date"], "book": book, "symbol": sh["symbol"], "side": "short", "notional": sh["notional"], "rate": sh["fee"], "day_count": cfg.day_count, "amount": -sh["notional"] * sh["fee"] / cfg.day_count, "kind": "borrow_fee"})
    reb = pd.DataFrame({"date": sh["date"], "book": book, "symbol": sh["symbol"], "side": "short", "notional": sh["notional"], "rate": sh["effr"], "day_count": cfg.day_count, "amount": sh["notional"] * sh["effr"] / cfg.day_count, "kind": "short_rebate"})
    d = daily.reset_index(); rd = effr.reindex(d["date"], method="ffill").values
    cash_rate = np.where(d["cash"] < 0, rd + cfg.financing_spread, rd + cfg.cash_spread)
    cash = pd.DataFrame({"date": d["date"], "book": book, "symbol": "CASH", "side": "cash", "notional": d["cash"], "rate": cash_rate, "day_count": cfg.day_count, "amount": d["cash"] * cash_rate / cfg.day_count, "kind": "cash_interest"})
    out = pd.concat([fee, reb, cash], ignore_index=True).sort_values(["date", "kind", "symbol"]).reset_index(drop=True)
    return out


# ---- the prime broker's version --------------------------------------------------------------------------------------------
SEEDS = ("stale_fee_rate", "rebate_act365", "missing_rebate", "duplicate_fee", "collateral_102", "stale_mark", "old_symbol", "buy_in_charge", "stale_effr")


def pb_statement(book: pd.DataFrame, positions: pd.DataFrame, events: pd.DataFrame | None = None, renames: pd.DataFrame | None = None, cfg: FinConfig | None = None) -> pd.DataFrame:
    """the same accruals as the PB would report them, with nine kinds of seeded discrepancy tagged in a hidden column"""
    cfg = cfg or FinConfig(); rng = np.random.default_rng(cfg.seed)
    pb = book.copy(); pb["seeded"] = ""
    pb["line_id"] = [f"PB{i:08d}" for i in range(len(pb))]
    dates = np.array(sorted(pb["date"].unique()))
    is_fee = pb["kind"] == "borrow_fee"; is_reb = pb["kind"] == "short_rebate"
    # 1. stale fee rate: for 1.5 % of (symbol, month) fee groups the PB carries the previous month's rate for the whole month
    pb["ym"] = pb["date"].dt.to_period("M")
    groups = pb[is_fee].groupby(["symbol", "ym"]).size().index.tolist()
    pick = [groups[i] for i in rng.choice(len(groups), size=max(1, int(0.015 * len(groups))), replace=False)]
    for sym, ym in pick:
        cur = is_fee & (pb["symbol"] == sym) & (pb["ym"] == ym)
        prev = is_fee & (pb["symbol"] == sym) & (pb["ym"] == ym - 1)
        if prev.any():
            old = float(pb.loc[prev, "rate"].iloc[-1])
            if abs(old - float(pb.loc[cur, "rate"].iloc[0])) > 1e-6:
                pb.loc[cur, "rate"] = old; pb.loc[cur, "amount"] = -pb.loc[cur, "notional"] * old / cfg.day_count; pb.loc[cur, "seeded"] = "stale_fee_rate"
    # 2. ACT/365 on the rebate for two symbols over one quarter each
    for sym in rng.choice(pb.loc[is_reb, "symbol"].unique(), size=2, replace=False):
        q = rng.choice(pb.loc[is_reb & (pb["symbol"] == sym), "date"].dt.to_period("Q").unique())
        m = is_reb & (pb["symbol"] == sym) & (pb["date"].dt.to_period("Q") == q)
        pb.loc[m, "day_count"] = 365; pb.loc[m, "amount"] = pb.loc[m, "notional"] * pb.loc[m, "rate"] / 365; pb.loc[m, "seeded"] = "rebate_act365"
    # 3. missing rebate lines: 0.3 % of them
    idx = pb.index[is_reb]; drop = rng.choice(idx, size=max(1, int(0.003 * len(idx))), replace=False)
    missing = pb.loc[drop, ["date", "symbol", "kind"]].copy(); missing["seeded"] = "missing_rebate"
    pb = pb.drop(index=drop)
    # 4. duplicated fee lines: 0.2 %
    idx = pb.index[pb["kind"] == "borrow_fee"]; dup = rng.choice(idx, size=max(1, int(0.002 * len(idx))), replace=False)
    d2 = pb.loc[dup].copy(); d2["seeded"] = "duplicate_fee"; d2["line_id"] = d2["line_id"] + "D"; pb.loc[dup, "seeded"] = "duplicate_fee"
    pb = pd.concat([pb, d2], ignore_index=True)
    # 5. fee on 102 % collateral for one calendar month (a convention, not an error, but it has to be explained)
    ym = rng.choice(pb.loc[pb["kind"] == "borrow_fee", "ym"].unique())
    m = (pb["kind"] == "borrow_fee") & (pb["ym"] == ym) & (pb["seeded"] == "")
    pb.loc[m, "notional"] = pb.loc[m, "notional"] * cfg.pb_collateral; pb.loc[m, "amount"] = -pb.loc[m, "notional"] * pb.loc[m, "rate"] / cfg.day_count; pb.loc[m, "seeded"] = "collateral_102"
    # 6. stale mark: 0.5 % of fee lines use the previous day's close for the notional
    pos = positions.copy(); pos["date"] = pd.to_datetime(pos["date"]); prev_close = pos.sort_values("date").groupby("symbol")["close"].shift(1); pos["prev_close"] = prev_close
    pc = pos.set_index(["date", "symbol"])["prev_close"]
    idx = pb.index[(pb["kind"] == "borrow_fee") & (pb["seeded"] == "")]; st = rng.choice(idx, size=max(1, int(0.005 * len(idx))), replace=False)
    for i in st:
        key = (pb.at[i, "date"], pb.at[i, "symbol"])
        if key in pc.index and pd.notna(pc[key]):
            cl = pos.set_index(["date", "symbol"])["close"].get(key)
            if cl and abs(pc[key] / cl - 1) > 1e-4:
                pb.at[i, "notional"] = pb.at[i, "notional"] * pc[key] / cl; pb.at[i, "amount"] = -pb.at[i, "notional"] * pb.at[i, "rate"] / cfg.day_count; pb.at[i, "seeded"] = "stale_mark"
    # 7. the old ticker for the first three days after a rename
    if renames is not None and len(renames):
        for r in renames.itertuples():
            vf = pd.Timestamp(r.valid_from); m = (pb["symbol"] == r.symbol) & (pb["date"] >= vf) & (pb["date"] < vf + pd.Timedelta(days=5))
            if m.any():
                pb.loc[m, "symbol"] = r.old_symbol; pb.loc[m, "seeded"] = "old_symbol"
    # 8. buy-in charges on a few recalls: a line the book does not have
    if events is not None and len(events):
        rec = events[events["event"] == "recall"]
        if len(rec):
            take = rec.iloc[rng.choice(len(rec), size=min(len(rec), 6), replace=False)]
            bi = pd.DataFrame({"date": pd.to_datetime(take["date"]), "book": book["book"].iloc[0], "symbol": take["symbol"], "side": "short", "notional": take["notional"], "rate": 0.0, "day_count": cfg.day_count, "amount": -0.001 * take["notional"], "kind": "buy_in", "seeded": "buy_in_charge", "ym": pd.to_datetime(take["date"]).dt.to_period("M"), "line_id": [f"PBBI{i:04d}" for i in range(len(take))]})
            pb = pd.concat([pb, bi], ignore_index=True)
    # 9. yesterday's EFFR on the day it changes (the PB posts the rate with a one-day lag) for one rate-change day
    reb = pb[(pb["kind"] == "short_rebate") & (pb["seeded"] == "")]
    daily_rate = reb.groupby("date")["rate"].first(); changes = daily_rate[daily_rate.diff().abs() > 1e-6]
    if len(changes):
        d = changes.index[rng.integers(0, len(changes))]; i = list(daily_rate.index).index(d); old = float(daily_rate.iloc[i - 1])
        m = (pb["kind"] == "short_rebate") & (pb["date"] == d) & (pb["seeded"] == "")
        pb.loc[m, "rate"] = old; pb.loc[m, "amount"] = pb.loc[m, "notional"] * old / pb.loc[m, "day_count"]; pb.loc[m, "seeded"] = "stale_effr"
    pb = pb.drop(columns=["ym"]).sort_values(["date", "kind", "symbol"]).reset_index(drop=True)
    pb.attrs["missing"] = missing
    return pb


# ---- reconciliation ----------------------------------------------------------------------------------------------------------
def reconcile(book: pd.DataFrame, pb: pd.DataFrame, renames: pd.DataFrame | None = None, tol_usd: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """three-way check per (date, symbol, kind): presence, line count, rate, notional, day count, amount.  Returns the breaks
    (one per key) and the score against the seeded discrepancies, which the reconciliation itself never sees."""
    key = ["date", "symbol", "kind"]
    b = book.groupby(key).agg(n_book=("amount", "size"), amount_book=("amount", "sum"), rate_book=("rate", "first"), notional_book=("notional", "first"), dc_book=("day_count", "first")).reset_index()
    p = pb.groupby(key).agg(n_pb=("amount", "size"), amount_pb=("amount", "sum"), rate_pb=("rate", "first"), notional_pb=("notional", "first"), dc_pb=("day_count", "first"), seeded=("seeded", lambda s: ",".join(sorted(set(x for x in s if x))))).reset_index()
    m = b.merge(p, on=key, how="outer", indicator="src")
    old_of = {}; new_of = {}
    if renames is not None and len(renames):
        old_of = dict(zip(renames["symbol"], renames["old_symbol"])); new_of = dict(zip(renames["old_symbol"], renames["symbol"]))
    pb_keys = set(zip(p["date"], p["symbol"], p["kind"])); book_keys = set(zip(b["date"], b["symbol"], b["kind"]))
    rows = []
    for r in m.itertuples(index=False):
        if r.src == "left_only":
            if r.symbol in old_of and (r.date, old_of[r.symbol], r.kind) in pb_keys:
                rows.append((r.date, r.symbol, r.kind, "symbol_mapping", f"PB carries {old_of[r.symbol]} for {r.symbol}", float(r.amount_book), r.seeded if isinstance(r.seeded, str) else ""))
            else:
                rows.append((r.date, r.symbol, r.kind, "missing_in_pb", f"book {r.amount_book:.2f}, no PB line", float(r.amount_book), ""))
        elif r.src == "right_only":
            if r.symbol in new_of and (r.date, new_of[r.symbol], r.kind) in book_keys:
                continue        # the other side of the mapping break, reported once from the book side
            if r.kind not in KINDS:
                rows.append((r.date, r.symbol, r.kind, "unknown_charge", f"PB {r.kind} {r.amount_pb:.2f} with no book line", float(r.amount_pb), r.seeded))
            else:
                rows.append((r.date, r.symbol, r.kind, "missing_in_book", f"PB {r.amount_pb:.2f}, no book line", float(r.amount_pb), r.seeded))
        else:
            if r.n_pb > r.n_book:
                rows.append((r.date, r.symbol, r.kind, "duplicate_line", f"{int(r.n_pb)} PB lines vs {int(r.n_book)}", float(r.amount_pb - r.amount_book), r.seeded)); continue
            diff = float(r.amount_pb - r.amount_book)
            # the terms (rate, notional basis, day count) are compared exactly: a small dollar difference on a general-collateral
            # name is still a wrong term, and wrong terms are systematic; the dollar tolerance only applies to the residual
            if abs(r.rate_pb - r.rate_book) > 1e-9:
                rows.append((r.date, r.symbol, r.kind, "rate_break", f"PB rate {r.rate_pb:.6f} vs book {r.rate_book:.6f}", diff, r.seeded))
            elif r.notional_book and abs(r.notional_pb / r.notional_book - 1) > 1e-6:
                ratio = r.notional_pb / r.notional_book
                kind = "collateral_basis" if abs(ratio - 1.02) < 1e-4 else "notional_break"
                rows.append((r.date, r.symbol, r.kind, kind, f"PB notional {r.notional_pb:,.0f} vs book {r.notional_book:,.0f} (x{ratio:.4f})", diff, r.seeded))
            elif r.dc_pb != r.dc_book or (r.notional_book and r.rate_book and r.amount_pb and abs(abs(r.notional_book * r.rate_book / r.amount_pb) - r.dc_book) > 0.5):
                rows.append((r.date, r.symbol, r.kind, "day_count_break", f"PB day count {r.dc_pb} vs book {r.dc_book}", diff, r.seeded))
            elif abs(diff) > tol_usd:
                rows.append((r.date, r.symbol, r.kind, "amount_break", f"PB {r.amount_pb:.2f} vs book {r.amount_book:.2f}", diff, r.seeded))
    breaks = pd.DataFrame(rows, columns=["date", "symbol", "kind", "break_type", "detail", "amount_usd", "seeded"])
    breaks["material"] = breaks["amount_usd"].abs() > tol_usd
    # score: every seeded key should produce a break of the matching type; no unseeded key should
    expected = {"stale_fee_rate": "rate_break", "rebate_act365": "day_count_break", "missing_rebate": "missing_in_pb", "duplicate_fee": "duplicate_line", "collateral_102": "collateral_basis", "stale_mark": "notional_break", "old_symbol": "symbol_mapping", "buy_in_charge": "unknown_charge", "stale_effr": "rate_break"}
    seeded_keys = pb[pb["seeded"] != ""].groupby(["date", "symbol", "kind"])["seeded"].first()
    missing = pb.attrs.get("missing")
    score = {}
    found = breaks.set_index(["date", "symbol", "kind"])["break_type"].to_dict() if len(breaks) else {}
    found_mapped = {(d, new_of.get(s, s), k): v for (d, s, k), v in found.items()}
    for seed, bt in expected.items():
        if seed == "missing_rebate":
            keys = [(r.date, r.symbol, r.kind) for r in missing.itertuples()] if missing is not None else []
        elif seed == "old_symbol":
            keys = [(d, new_of.get(s, s), k) for (d, s, k), v in seeded_keys.items() if v == seed]
        else:
            keys = [k for k, v in seeded_keys.items() if v == seed]
        n = len(keys); hit = sum(1 for k in keys if found_mapped.get(k) == bt or found.get(k) == bt); other = sum(1 for k in keys if (found_mapped.get(k) or found.get(k)) not in (None, bt))
        score[seed] = {"seeded": n, "detected_as_expected": hit, "detected_other_type": other, "expected_type": bt}
    n_unseeded = int((breaks["seeded"] == "").sum()) if len(breaks) else 0
    n_unseeded -= int(((breaks["seeded"] == "") & (breaks["break_type"].isin(["missing_in_pb", "symbol_mapping"]))).sum()) if len(breaks) else 0   # those carry no PB row to be tagged
    summary = {"book_lines": int(len(book)), "pb_lines": int(len(pb)), "keys": int(len(m)), "breaks": int(len(breaks)), "material_breaks": int(breaks["material"].sum()) if len(breaks) else 0, "breaks_by_type": breaks["break_type"].value_counts().to_dict() if len(breaks) else {}, "breaks_usd_by_type": breaks.groupby("break_type")["amount_usd"].sum().round(2).to_dict() if len(breaks) else {},
               "seeded": score, "false_positives": n_unseeded, "tolerance_usd": tol_usd,
               "overall_detection": float(sum(v["detected_as_expected"] for v in score.values()) / max(sum(v["seeded"] for v in score.values()), 1))}
    return breaks, summary


def financing_summary(acc: pd.DataFrame, nav: pd.Series, n_days: int) -> dict:
    """annualised accruals as a share of the running NAV (nav: a Series indexed by date), by kind"""
    a = acc.copy(); a["nav"] = nav.reindex(pd.to_datetime(a["date"])).values; a["frac"] = a["amount"] / a["nav"]
    tot = a.groupby("kind")["frac"].sum()
    ann = {k: float(v * 252 / max(n_days, 1)) for k, v in tot.items()}
    usd = a.groupby("kind")["amount"].sum()
    return {"annualised_over_nav": ann, "net_financing_ann": float(sum(ann.values())), "usd_total": {k: float(v) for k, v in usd.items()}, "lines": int(len(acc)), "days": int(acc["date"].nunique())}
