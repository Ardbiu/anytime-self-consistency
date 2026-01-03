#!/bin/bash
set -e

echo "=== STARTING SMOKE TEST ==="

mkdir -p outputs/logs
TMP_LOG_FILE="outputs/logs/smoke_tmp.log"

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 1. Run Tiny Experiment
echo "[1/4] Running Eval..."
python -m src.run_eval --config configs/gsm8k_smoke.yaml > "$TMP_LOG_FILE" 2>&1 || { cat "$TMP_LOG_FILE"; exit 1; }

# Resolve run_id for log naming
RUN_ID=$(grep -m1 "Global Run ID:" "$TMP_LOG_FILE" | sed -E 's/.*Global Run ID: //')
if [ -z "$RUN_ID" ]; then
    latest_file=$(ls -t outputs/runs/*.jsonl 2>/dev/null | head -n 1)
    if [ -n "$latest_file" ]; then
        RUN_ID=$(basename "$latest_file" | sed -E 's/.*([0-9]{8}-[0-9]{6}_[0-9a-f]{6}).*/\1/')
    fi
fi
if [ -z "$RUN_ID" ]; then
    echo "FAIL: Could not determine run_id for smoke log naming."
    echo "See $TMP_LOG_FILE for details."
    exit 1
fi

LOG_FILE="outputs/logs/smoke_${RUN_ID}.log"
mv "$TMP_LOG_FILE" "$LOG_FILE"

# Check for forbidden warnings
if grep -q "generation flags are not valid and may be ignored" "$LOG_FILE"; then
    echo "FAIL: Found forbidden warning in logs:"
    grep "generation flags are not valid and may be ignored" "$LOG_FILE"
    echo "See $LOG_FILE for details."
    exit 1
fi
if grep -q "ignored flags" "$LOG_FILE"; then
    echo "FAIL: Found 'ignored flags' warning in logs."
    echo "See $LOG_FILE for details."
    exit 1
fi
if grep -q "Traceback" "$LOG_FILE"; then
    echo "FAIL: Found traceback in logs."
    echo "See $LOG_FILE for details."
    exit 1
fi

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
if [ ! -f "outputs/logs/smoke_${RUN_ID}.log" ]; then
    echo "FAIL: Smoke log not found."
    echo "Expected: outputs/logs/smoke_${RUN_ID}.log"
    exit 1
fi
if [ ! -f "outputs/summaries/smoke_summary.csv" ]; then
    echo "FAIL: Summary CSV not found."
    echo "See $LOG_FILE for details."
    exit 1
fi
if [ ! -f "outputs/plots/smoke_pareto.png" ]; then
    echo "FAIL: Pareto PNG not found."
    echo "See $LOG_FILE for details."
    exit 1
fi

echo "=== SMOKE TEST PASSED ==="
