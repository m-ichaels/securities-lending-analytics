"""Accruals, the prime-broker statement's seeded discrepancies and the reconciliation that has to find them."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slb import financing as F  # noqa: E402


@pytest.fixture
def book():
    """40 shorts over 130 business days with fees that change monthly, one rename, a few recalls"""
    rng = np.random.default_rng(3); days = pd.bdate_range("2024-01-02", periods=130); syms = [f"S{i:02d}" for i in range(40)]
    rows = []; fee_rows = []
    for s in syms:
        base = rng.choice([0.003, 0.004, 0.02, 0.08])
        for d in days:
            fee_rows.append((d, s, base * (1 + 0.1 * (d.month % 3)))); rows.append((d, s, -10000.0, 50.0 + rng.normal(0, 1)))
    for d in days:
        rows.append((d, "LONG", 20000.0, 100.0))
    pos = pd.DataFrame(rows, columns=["date", "symbol", "shares", "close"])
    fee_w = pd.DataFrame(fee_rows, columns=["date", "symbol", "fee"]).pivot(index="date", columns="symbol", values="fee")
    effr = pd.Series(np.where(days < pd.Timestamp("2024-03-21"), 0.0533, 0.0508), index=days)
    daily = pd.DataFrame({"cash": 5e5, "nav": 2.5e6}, index=days); daily.index.name = "date"
    acc = F.book_accruals(pos, daily, fee_w, effr)
    events = pd.DataFrame({"date": days[[10, 40, 70, 100]], "symbol": ["S01", "S05", "S09", "S13"], "event": "recall", "notional": 5e5, "hazard": 0.01})
    renames = pd.DataFrame([{"old_symbol": "OLD3", "symbol": "S03", "cusip": "x", "valid_from": "2024-04-01"}])
    return acc, pos, events, renames


def test_accrual_amounts(book):
    acc, pos, events, renames = book
    a = acc[(acc["symbol"] == "S00") & (acc["date"] == pd.Timestamp("2024-01-02"))].set_index("kind")
    notional = 10000 * pos[(pos["symbol"] == "S00") & (pos["date"] == pd.Timestamp("2024-01-02"))]["close"].iloc[0]
    assert a.loc["borrow_fee", "amount"] == pytest.approx(-notional * a.loc["borrow_fee", "rate"] / 360)
    assert a.loc["short_rebate", "amount"] == pytest.approx(notional * 0.0533 / 360)
    c = acc[acc["kind"] == "cash_interest"].iloc[0]
    assert c["amount"] == pytest.approx(5e5 * (0.0533 - 0.001) / 360)
    assert set(acc["kind"]) == {"borrow_fee", "short_rebate", "cash_interest"} and len(acc) == 130 * (40 * 2 + 1)


def test_clean_statement_reconciles_with_no_breaks(book):
    acc, pos, events, renames = book
    pb = acc.copy(); pb["seeded"] = ""; pb["line_id"] = "x"; pb.attrs["missing"] = None
    breaks, score = F.reconcile(acc, pb, renames)
    assert len(breaks) == 0 and score["false_positives"] == 0


def test_seeded_discrepancies_are_found_and_classified(book):
    acc, pos, events, renames = book
    pb = F.pb_statement(acc, pos, events, renames)
    assert set(pb["seeded"].unique()) - {""} == set(F.SEEDS) - {"missing_rebate"} and len(pb.attrs["missing"]) > 0     # the missing lines are, by construction, not on the statement
    breaks, score = F.reconcile(acc, pb, renames)
    assert score["false_positives"] == 0
    for seed, v in score["seeded"].items():
        assert v["detected_other_type"] == 0, seed
        assert v["detected_as_expected"] >= 0.9 * v["seeded"], (seed, v)         # the remainder are below the $0.50 tolerance
    assert score["overall_detection"] > 0.95
    assert set(breaks["break_type"]) >= {"rate_break", "day_count_break", "missing_in_pb", "duplicate_line", "collateral_basis", "symbol_mapping", "unknown_charge"}
    # the rename: the PB carries OLD3 for the first days of April, and the recon maps it rather than reporting two breaks
    sm = breaks[breaks["break_type"] == "symbol_mapping"]
    assert (sm["symbol"] == "S03").all() and (sm["date"] >= "2024-04-01").all() and not (breaks["symbol"] == "OLD3").any()


def test_financing_summary(book):
    acc, pos, events, renames = book
    s = F.financing_summary(acc, pd.Series(2.5e6, index=pd.bdate_range("2024-01-02", periods=130)), 130)
    assert s["lines"] == len(acc) and s["days"] == 130
    assert s["annualised_over_nav"]["short_rebate"] > 0 and s["annualised_over_nav"]["borrow_fee"] < 0
