"""slb command line.

  python -m slb build-store                 load the raw files into data/derived/slb.duckdb (prices, short interest, threshold, fails, symbol map, IB, EFFR)
  python -m slb borrow                      build the daily borrow table -> data/derived/borrow.parquet and the borrow table in the store
  python -m slb fee-check                   the modelled fee cross-section against the literature and the IB snapshots -> results/fee_check.json
  python -m slb backtest [--signal MOM|SIR|REV] [--quick]     the four books for one signal, printed
  python -m slb recon                       the financing layer and the prime-broker reconciliation on the headline book
  python -m slb run [--quick] [--reuse]     everything -> results/run.json and the daily series for the figures
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from . import data as D


def cmd_build_store(a):
    con = D.connect(fresh=True); D.build_store(con); con.close()


def cmd_borrow(a):
    from .run import build_borrow
    build_borrow(start=a.start)


def cmd_fee_check(a):
    from .run import RESULTS, fee_checks, load_borrow, to_json, wide_prices
    borrow = load_borrow(); con = D.connect(); px = con.execute("select symbol, date, close, adjclose from prices").df(); con.close()
    adj, _ = wide_prices(px); out = fee_checks(borrow, adj)
    print(json.dumps({k: v for k, v in out.items() if k != "monthly"}, indent=1, default=str)); to_json(out, os.path.join(RESULTS, "fee_check.json"))


def cmd_backtest(a):
    from .run import load_borrow, run_strategy, wide_prices
    borrow = load_borrow("2024-01-01" if a.quick else None); con = D.connect(); px = con.execute("select symbol, date, close, adjclose from prices").df(); con.close()
    adj, close = wide_prices(px); r = run_strategy(a.signal, borrow, adj, close, D.load_effr(), record_positions=False)
    print(pd.DataFrame(r["summary"]["drag_table"]).T.round(4).to_string()); print(json.dumps(r["summary"].get("recovery"), indent=1))


def cmd_recon(a):
    from . import backtest as BT, strategy as S
    from .run import HEADLINE, load_borrow, run_financing, wide_prices
    borrow = load_borrow("2024-01-01" if a.quick else None); con = D.connect(); px = con.execute("select symbol, date, close, adjclose from prices").df()
    adj, close = wide_prices(px); effr = D.load_effr(); renames = con.execute("select * from symbol_history").df()
    rd = S.month_ends(pd.DatetimeIndex(sorted(borrow["date"].unique())))[:-1]; naive = S.naive_targets(adj, borrow, rd, HEADLINE)
    res = BT.run_backtest(naive, adj, close, borrow, effr)
    out = run_financing(res, borrow, effr, con, renames); con.close()
    print(json.dumps(out, indent=1, default=str))


def cmd_run(a):
    from .run import run_all
    run_all(quick=a.quick, reuse=a.reuse)


def main(argv=None):
    p = argparse.ArgumentParser(prog="slb"); sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-store").set_defaults(fn=cmd_build_store)
    b = sub.add_parser("borrow"); b.add_argument("--start", default="2019-06-01"); b.set_defaults(fn=cmd_borrow)
    sub.add_parser("fee-check").set_defaults(fn=cmd_fee_check)
    bt = sub.add_parser("backtest"); bt.add_argument("--signal", default="MOM"); bt.add_argument("--quick", action="store_true"); bt.set_defaults(fn=cmd_backtest)
    rc = sub.add_parser("recon"); rc.add_argument("--quick", action="store_true"); rc.set_defaults(fn=cmd_recon)
    r = sub.add_parser("run"); r.add_argument("--quick", action="store_true"); r.add_argument("--reuse", action="store_true", help="use the store and data/derived/borrow.parquet as they are"); r.set_defaults(fn=cmd_run)
    a = p.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    main()
