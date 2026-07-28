#!/usr/bin/env python3
"""Writes notebooks/results.ipynb (a walkthrough of results/*.json, the figures and the DuckDB store); run it after the
pipeline so that the outputs are populated:  python scripts/make_notebook.py && jupyter nbconvert --execute --inplace notebooks/results.ipynb"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("# securities-lending-analytics: results walkthrough\n\nLoads `results/run.json`, the daily series and the DuckDB store produced by `scripts/run_all.sh`, and shows the figures and the headline numbers. Run the pipeline first (`scripts/run_all.sh --skip-download` works on the committed derived data).")
code("import json, os\nimport duckdb, pandas as pd\nfrom IPython.display import Image, display\nos.chdir(os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())\nrun = json.load(open('results/run.json', encoding='utf-8'))\npct = lambda x, d=2: f'{100 * x:.{d}f} %'\nprint('as of', run['as_of'], '|', run['universe'], 'names |', run['borrow_model'].get('rows', 0), 'name-days in the borrow table')")
md("## 1. The one number\n\nThe annual return drag from borrow fees and recalls on the headline book, and what the borrow-aware optimiser recovers.")
code("h = run['headline']; s = run['strategies'][h['strategy']]; bn = s['books']['naive']; ba = s['books']['borrow_aware']\nprint(f\"{h['strategy']} book, ${h['nav0']/1e6:.0f}m NAV, {h['gross_exposure']}, {s['rebalances']} rebalances {s['first']} .. {s['last']}\")\nprint(f\"naive: fees {pct(bn['ann_fees'])} + recalls {pct(bn['ann_recall_impact'])} = {pct(bn['ann_fees'] + bn['ann_recall_impact'])} of NAV a year; missed shorts {pct(bn['ann_opportunity_short'])}; locate failures {pct(bn['locate_failure_rate'], 1)}; {bn['recalls_per_month']:.1f} recalls a month\")\nprint(f\"borrow-aware: fees {pct(ba['ann_fees'])} + recalls {pct(ba['ann_recall_impact'])}; recovery vs the same optimiser without the borrow term {pct(h['drag_reduction_vs_risk_only'])} a year; net return {pct(ba['ann_return'])} vs {pct(bn['ann_return'])} naive\")\ndisplay(pd.DataFrame(s['drag_table']).T.round(4))\ndisplay(Image('results/figures/drag_waterfall.png'))")
md("## 2. The borrow model\n\nSpecialness score from the published short interest (with its publication lag), threshold-list membership, fails to deliver, dollar volume and price; the fee from the score through the literature's cross-section with a utilisation floor; lendable supply, utilisation, locate probability and recall hazard.")
code("fc = run['fee_checks']; d = fc['distribution_last_day']\nprint(json.dumps({k: v for k, v in d.items()}, indent=1))\nprint('threshold names:', fc['threshold_names'])\ndisplay(pd.DataFrame(fc['forward_return_by_fee_bucket']).T)\ndisplay(Image('results/figures/fee_cross_section.png'))\ndisplay(Image('results/figures/fee_returns.png'))")
code("con = duckdb.connect('data/derived/slb.duckdb', read_only=True)\ndisplay(con.execute(\"select symbol, round(fee*100,2) as fee_pct, round(sir*100,1) as sir_pct, days_to_cover, on_threshold, round(utilisation,2) as utilisation, round(p_locate,2) as p_locate, round(recall_hazard*1e4,1) as hazard_bp from borrow where date = (select max(date) from borrow) order by fee desc limit 15\").df())\ndisplay(con.execute('select * from symbol_history order by valid_from desc limit 10').df())\ncon.close()")
md("## 3. The three books under four treatments\n\nFrictionless, naive, the optimiser without the borrow term (isolates the alpha tilt) and the borrow-aware optimiser.")
code("rows = []\nfor n, s in run['strategies'].items():\n    for bk, v in s['drag_table'].items():\n        rows.append({'strategy': n, 'book': bk, **{k: round(x, 4) for k, x in v.items()}})\ndisplay(pd.DataFrame(rows).set_index(['strategy', 'book']))\ndisplay(Image('results/figures/strategies.png'))\ndisplay(Image('results/figures/nav.png'))")
md("## 4. Unavailability, recalls and margin")
code("reb = pd.read_csv('results/headline_rebalances.csv', parse_dates=['date'])\nprint('locate failures by rebalance (last 12):')\ndisplay(reb[['date', 'n_short_requested', 'n_locate_failed', 'failed', 'fee_wavg_short']].tail(12))\nev = pd.read_csv('results/headline_events.csv')\nprint(ev['event'].value_counts().to_dict()); print('most often unavailable:', ev[ev['event'] == 'locate_failed']['symbol'].value_counts().head(10).to_dict())\ndisplay(Image('results/figures/availability.png'))")
md("## 5. Sensitivity\n\nFee level, recall intensity, the random draws and the optimiser's alpha assumption.")
code("if 'sensitivity' in run:\n    display(pd.DataFrame(run['sensitivity']['grid']).round(4)); display(pd.DataFrame(run['sensitivity']['seeds']).round(4))\ndisplay(pd.DataFrame(run.get('alpha_sensitivity', [])).round(4))\ndisplay(Image('results/figures/sensitivity.png'))")
md("## 6. Financing and the prime-broker reconciliation\n\nAccruals ACT/360 (fee on the short notional, rebate on the proceeds, cash at EFFR plus a spread), Regulation T and portfolio margin, and the reconciliation of a PB statement with nine kinds of seeded discrepancy.")
code("f = run['financing']; rc = f['reconciliation']\nprint({k: v for k, v in f.items() if k not in ('reconciliation', 'margin_last_day')}); print(f['margin_last_day'])\ndisplay(pd.DataFrame(rc['seeded']).T); print(rc['breaks_by_type'], rc['breaks_usd_by_type'], 'false positives', rc['false_positives'], 'detection', rc['overall_detection'])\nbr = pd.read_csv('results/fin_breaks.csv'); display(br.sample(min(10, len(br)), random_state=1))\ndisplay(Image('results/figures/recon.png'))")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
with open(os.path.join(ROOT, "notebooks", "results.ipynb"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/results.ipynb")
