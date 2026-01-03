#!/bin/bash
set -e

echo "=== STARTING SMOKE TEST ==="

# 1. Run Tiny Experiment
echo "[1/4] Running Eval..."
python -m src.run_eval --config configs/gsm8k_smoke.yaml

# 2. Aggregate
echo "[2/4] Aggregating..."
python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/smoke_summary.csv --latest

# 3. Plot
echo "[3/4] Plotting..."
python scripts/plot_pareto.py --input outputs/summaries/smoke_summary.csv --output outputs/plots/smoke_pareto.png --latest

# 4. Diagnose
echo "[4/4] Diagnosing Sampling..."
python scripts/diagnose_sampling.py --latest

# 5. Assertions
echo "[5/5] Verify Artifacts..."
if [ ! -f "outputs/summaries/smoke_summary.csv" ]; then
    echo "FAIL: Summary CSV not found."
    exit 1
fi
if [ ! -f "outputs/plots/smoke_pareto.png" ]; then
    echo "FAIL: Pareto PNG not found."
    exit 1
fi

echo "=== SMOKE TEST PASSED ==="
