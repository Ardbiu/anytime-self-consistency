#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load python/3.10 || true
  fi
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Set PYTHON=/path/to/python or load a python module." >&2
  exit 1
fi

RUN_GROUP="${RUN_GROUP:-h200_3h_test_$(date +%m%d%H%M)}"
echo "Run group: $RUN_GROUP"

mkdir -p outputs/plots outputs/summaries outputs/logs

echo "Core method smoke (all methods, GSM8K)"
"$PYTHON_BIN" -m src.run_eval --config configs/suite_smoke_full.yaml --seed 0 --run_group "$RUN_GROUP"

echo "Multiple choice checks (ARC, MMLU)"
"$PYTHON_BIN" -m src.run_eval --config configs/arc_challenge_smoke.yaml --seed 0 --limit 30 --run_group "$RUN_GROUP"
"$PYTHON_BIN" -m src.run_eval --config configs/mmlu_smoke.yaml --seed 0 --limit 30 --run_group "$RUN_GROUP"

echo "Code eval checks (HumanEval, MBPP)"
"$PYTHON_BIN" -m src.run_eval --config configs/humaneval_smoke.yaml --seed 0 --limit 10 --run_group "$RUN_GROUP"
"$PYTHON_BIN" -m src.run_eval --config configs/mbpp_smoke.yaml --seed 0 --limit 20 --run_group "$RUN_GROUP"

echo "Aggregate + diagnostics"
"$PYTHON_BIN" scripts/aggregate_results.py --run_group "$RUN_GROUP" --bootstrap 200 --prompt_cost 0.05 --completion_cost 1.0
"$PYTHON_BIN" scripts/plot_pareto.py --run_group "$RUN_GROUP" --grouped
"$PYTHON_BIN" scripts/plot_pareto.py --run_group "$RUN_GROUP" --grouped --x_metric weighted
"$PYTHON_BIN" scripts/analyze_stopping_bounds.py --run_group "$RUN_GROUP" --output_plot "outputs/plots/stopping_risk_${RUN_GROUP}.png"
"$PYTHON_BIN" scripts/pareto_dominance.py --run_group "$RUN_GROUP" --summary_output "outputs/summaries/pareto_summary_${RUN_GROUP}.csv"
"$PYTHON_BIN" scripts/significance_tests.py --run_group "$RUN_GROUP" --dataset gsm8k \
  --method_a anytime_sc --method_b self_consistency --a_budget 256 --a_delta 0.1 --b_budget 256
"$PYTHON_BIN" scripts/plot_confidence_trajectory.py --run_group "$RUN_GROUP" --dataset gsm8k --method anytime_sc \
  --output "outputs/plots/confidence_${RUN_GROUP}.png"

echo "Done. Run group: $RUN_GROUP"
