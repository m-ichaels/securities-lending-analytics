#!/usr/bin/env python3
"""report.pdf from results/summary.md and results/figures/*.png (fpdf2).   python scripts/report.py [results] [report.pdf]"""
import os
import sys

from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "report.pdf")

INTRO = """Question. A systematic long-short equity book pays to borrow every share it is short, loses a short when the lender recalls it, and sometimes cannot find the stock at all. How much of the book's return goes to borrow fees and recalls, how often does a short leg become unavailable, and how much of that does a portfolio optimiser that knows the borrow cost get back? This stack answers with a borrow-cost database built from the free public record (FINRA bi-monthly short interest, the Nasdaq and NYSE Regulation SHO threshold lists, the SEC's fails-to-deliver files, a point-in-time ticker map from the CUSIPs in those files, shares outstanding and the effective federal funds rate), a daily simulation of locates, recalls, fees, rebates and margin on three monthly long-short books, a borrow-aware optimiser, and a financing layer with a simulated prime-broker statement reconciled against the book's own accruals.

Method. Python package (slb) on DuckDB and parquet. The borrow model turns what a desk can observe on each day (published short interest and days to cover with FINRA's nine-business-day lag, threshold-list membership from the next morning, fails to deliver a month later, dollar volume and price) into a specialness score, maps the score to an annual fee through the published cross-section of fees (D'Avolio 2002; Engelberg, Reed and Ringgenberg 2018; Muravyev, Pearson and Pollet 2022), with a utilisation floor (Kolasinski, Reed and Ringgenberg 2013), and derives lendable supply, utilisation, a locate probability and a daily recall hazard. Three signals (12-1 momentum, published short interest, one-month reversal) build equal-weighted decile books, 100 % long and 100 % short of a $500m NAV, rebalanced monthly. Each book runs four ways: with no lending frictions, naive, through the optimiser without the borrow term, and through the borrow-aware optimiser. A shadow short leg holds every intended short so the return the book missed is measured rather than assumed. The financing layer accrues fees, rebates and cash interest ACT/360, computes Regulation T and portfolio margin, and reconciles a prime-broker statement with nine kinds of seeded discrepancy.

Caveats. Interactive Brokers' shortable file (the one free source of actual fees) could not be reached from this network or from GitHub's runners, so the fees are modelled and the fee level is a stated assumption checked by sensitivity; the code fits the map to IB snapshots when the fetch workflow delivers them. The universe is selected on the latest FINRA file, so it is heavily shorted by construction and survivorship-biased; strategy returns are context, the drag is the object of study."""

FIGS = [("fee_cross_section.png", "The borrow model: the modelled fee cross-section on the last day against the published quantiles, fee against published short interest with the threshold-list names marked, and the universe's short interest, utilisation and specials through time."),
        ("drag_waterfall.png", "The one number: annualised decomposition of the headline book from gross price P&L to net, naive and borrow-aware, with the drag from borrow and recalls."),
        ("nav.png", "The headline book under the four treatments; what the naive book paid in fees and recalls; the short leg's weighted fee and the intended shorts it could not hold."),
        ("strategies.png", "The three books: drag components naive against borrow-aware, and net returns."),
        ("availability.png", "Unavailability: locates failed at each rebalance, recalls per month, and the margin headroom under Regulation T and portfolio margin."),
        ("sensitivity.png", "Fee level, recall intensity and random-draw sensitivity of the drag and of the optimiser's recovery."),
        ("fee_returns.png", "A check on the fee ordering: next-month returns by fee bucket (expensive-to-short names should underperform), and the fee level through time."),
        ("recon.png", "The prime-broker reconciliation: seeded discrepancies found by type, breaks by type and their dollar size.")]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9); self.set_text_color(120); self.cell(0, 6, "securities-lending-analytics - borrow-cost model, locates and recalls, borrow-aware optimiser, financing and reconciliation", align="R"); self.ln(8); self.set_text_color(0)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "", 8); self.set_text_color(120); self.cell(0, 6, f"{self.page_no()}", align="C")


def clean(s):
    return (s.replace("–", "-").replace("—", "-").replace("−", "-").replace("×", "x").replace("≥", ">=").replace("≤", "<=").replace("…", "...").replace("²", "^2").replace("±", "+/-").replace("**", "").replace("`", "")
             .replace("→", "->").replace("≈", "~").replace("é", "e").replace("ö", "o").replace("’", "'").replace("λ", "lambda").replace("σ", "sigma").replace("α", "alpha"))


def md_table(pdf, rows):
    cols = [c.strip() for c in rows[0].strip("|").split("|")]
    data = [[clean(c.strip()) for c in r.strip("|").split("|")] for r in rows[2:]]
    n = len(cols); w = (pdf.w - 20) / n; fs = 6.5 if n <= 7 else 5.0 if n <= 12 else 4.2; cut = 42 if n <= 7 else 22 if n <= 12 else 14
    pdf.set_font("Helvetica", "B", fs)
    for c in cols:
        pdf.cell(w, 5, clean(c)[:cut], border=1)
    pdf.ln(5); pdf.set_font("Helvetica", "", fs)
    for r in data[:80]:
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
        for c in r:
            pdf.cell(w, 4.5, c[:cut], border=1)
        pdf.ln(4.5)
    pdf.ln(2)


def main():
    pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, "Securities Lending and Financing Analytics for a Long-Short Book", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for para in INTRO.split("\n\n"):
        pdf.multi_cell(0, 4.5, clean(para)); pdf.ln(2)
    for fn, cap in FIGS:
        p = os.path.join(R, "figures", fn)
        if not os.path.exists(p):
            continue
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()
        pdf.image(p, w=pdf.w - 20); pdf.set_font("Helvetica", "I", 8); pdf.multi_cell(0, 4, clean(cap)); pdf.ln(3); pdf.set_font("Helvetica", "", 9)
    sm = os.path.join(R, "summary.md")
    if os.path.exists(sm):
        pdf.add_page(); lines = open(sm, encoding="utf-8").read().splitlines(); i = 0
        while i < len(lines):
            l = lines[i]
            if l.startswith("## "):
                pdf.set_font("Helvetica", "B", 11); pdf.ln(2); pdf.cell(0, 7, clean(l[3:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("### "):
                pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 6, clean(l[4:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("|"):
                j = i
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                if j - i >= 2:
                    md_table(pdf, lines[i:j])
                i = j
            elif l.startswith("# "):
                i += 1
            elif l.strip():
                pdf.set_x(pdf.l_margin); pdf.multi_cell(0, 4.5, clean(l.strip())); i += 1
            else:
                i += 1
    pdf.output(OUT); print("wrote", OUT)


if __name__ == "__main__":
    main()
