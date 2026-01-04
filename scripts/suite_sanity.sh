#!/bin/bash
set -e

echo "=== STARTING SUITE SANITY TEST ==="

mkdir -p outputs/logs
TMP_LOG_FILE="outputs/logs/suite_sanity_tmp.log"

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python scripts/run_suite.py --config configs/suite_smoke.yaml --seeds 0,1 --datasets gsm8k,gsm_plus > "$TMP_LOG_FILE" 2>&1 || { cat "$TMP_LOG_FILE"; exit 1; }

RUN_GROUP=$(grep -m1 "Run group:" "$TMP_LOG_FILE" | sed -E 's/.*Run group: //')
if [ -z "$RUN_GROUP" ]; then
    latest_file=$(ls -t outputs/runs/*.jsonl 2>/dev/null | head -n 1)
    if [ -n "$latest_file" ]; then
        RUN_GROUP=$(basename "$latest_file" | sed -E 's/.*_([0-9]{8}-[0-9]{6})_seed.*/\1/')
    fi
fi
if [ -z "$RUN_GROUP" ]; then
    echo "FAIL: Could not determine run_group."
    echo "See $TMP_LOG_FILE for details."
    exit 1
fi

LOG_FILE="outputs/logs/suite_sanity_${RUN_GROUP}.log"
mv "$TMP_LOG_FILE" "$LOG_FILE"

if grep -q "generation flags are not valid and may be ignored" "$LOG_FILE"; then
    echo "FAIL: Found forbidden warning in logs."
    echo "See $LOG_FILE for details."
    exit 1
fi
if grep -q "Traceback" "$LOG_FILE"; then
    echo "FAIL: Found traceback in logs."
    echo "See $LOG_FILE for details."
    exit 1
fi

python scripts/aggregate_results.py --run_group "$RUN_GROUP" --bootstrap 200
python scripts/plot_pareto.py --run_group "$RUN_GROUP" --grouped
python scripts/diagnose_sampling.py --run_group "$RUN_GROUP"

if [ ! -f "outputs/summaries/summary_per_run.csv" ]; then
    echo "FAIL: summary_per_run.csv not found."
    exit 1
fi
if [ ! -f "outputs/summaries/summary_grouped.csv" ]; then
    echo "FAIL: summary_grouped.csv not found."
    exit 1
fi
if [ ! -f "outputs/plots/pareto_grouped.png" ]; then
    echo "FAIL: pareto_grouped.png not found."
    exit 1
fi

python - <<PY
import sys
import pandas as pd

per_run = pd.read_csv("outputs/summaries/summary_per_run.csv")
expected_limit = 10
if "count" not in per_run.columns:
    sys.exit("FAIL: summary_per_run.csv missing count column.")
if (per_run["count"] != expected_limit).any():
    bad = per_run[per_run["count"] != expected_limit]
    sys.exit(f"FAIL: count != {expected_limit} in summary_per_run.csv:\\n{bad}")

grouped = pd.read_csv("outputs/summaries/summary_grouped.csv")
required_cols = {"accuracy_ci_low", "accuracy_ci_high", "tokens_ci_low", "tokens_ci_high"}
missing = required_cols - set(grouped.columns)
if missing:
    sys.exit(f"FAIL: summary_grouped.csv missing CI columns: {sorted(missing)}")
PY

echo "=== SUITE SANITY TEST PASSED ==="
