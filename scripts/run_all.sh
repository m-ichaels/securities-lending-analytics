#!/usr/bin/env bash
# Full pipeline: raw data -> store -> borrow table -> the three books under four treatments, sensitivities, the financing
# layer and the reconciliation -> tests -> figures -> summary -> report.   Usage: scripts/run_all.sh [--skip-download] [--quick]
set -euo pipefail
Q=""
[[ " $* " == *" --quick "* ]] && Q="--quick"
[[ " $* " == *" --skip-download "* ]] || python tools/download.py all --from 2019-01-01
python -m slb build-store
python -m slb borrow
python -m slb fee-check
python -m slb run $Q --reuse | tee results/run.log
python -m pytest -q | tee results/tests.txt
python scripts/plots.py
python scripts/summarize.py
python scripts/report.py
echo done
