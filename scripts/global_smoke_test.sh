#!/bin/bash
set -e

echo "=== STARTING GLOBAL SMOKE TEST ==="

mkdir -p outputs/logs outputs/tmp
TMP_LOG_FILE="outputs/logs/global_smoke_tmp.log"
CFG_FILE="outputs/tmp/global_smoke.yaml"

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python - <<'PY'
import yaml

cfg = yaml.safe_load(open("configs/gsm8k_smoke.yaml", "r"))
cfg["limit"] = 20
cfg["methods"] = [
    {
        "name": "global_anytime_sc",
        "policy": "cot",
        "global_budget_tokens": [2000, 4000],
        "init_k": 1,
        "allocation_policy": ["uniform", "uncertainty", "voi", "ucb", "random", "finish_one", "per_example_budget"],
        "max_samples_per_item": 5,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 50,
        "ucb_c": 1.0,
    }
]
yaml.safe_dump(cfg, open("outputs/tmp/global_smoke.yaml", "w"))
print("Wrote outputs/tmp/global_smoke.yaml")
PY

echo "[1/3] Running Global Eval..."
python -m src.run_eval --config "$CFG_FILE" > "$TMP_LOG_FILE" 2>&1 || { cat "$TMP_LOG_FILE"; exit 1; }

RUN_ID=$(grep -m1 "Global Run ID:" "$TMP_LOG_FILE" | sed -E 's/.*Global Run ID: //')
if [ -z "$RUN_ID" ]; then
    latest_file=$(ls -t outputs/runs/*.jsonl 2>/dev/null | head -n 1)
    if [ -n "$latest_file" ]; then
        RUN_ID=$(basename "$latest_file" | sed -E 's/.*([0-9]{8}-[0-9]{6}_[0-9a-f]{6}).*/\1/')
    fi
fi
if [ -z "$RUN_ID" ]; then
    echo "FAIL: Could not determine run_id for global smoke log naming."
    echo "See $TMP_LOG_FILE for details."
    exit 1
fi

LOG_FILE="outputs/logs/global_smoke_${RUN_ID}.log"
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

echo "[2/3] Aggregating..."
python scripts/aggregate_results.py --run_id "$RUN_ID" --bootstrap 50

echo "[3/3] Plotting..."
python scripts/plot_global_curve.py --output outputs/plots/global_curve.png

if [ ! -f "outputs/summaries/summary_global_points.csv" ]; then
    echo "FAIL: summary_global_points.csv not found."
    exit 1
fi
if [ ! -f "outputs/summaries/summary_global_curve.csv" ]; then
    echo "FAIL: summary_global_curve.csv not found."
    exit 1
fi
if [ ! -f "outputs/plots/global_curve.png" ]; then
    echo "FAIL: global_curve.png not found."
    exit 1
fi

echo "=== GLOBAL SMOKE TEST PASSED ==="
