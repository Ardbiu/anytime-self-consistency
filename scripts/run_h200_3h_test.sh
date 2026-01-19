#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

RUN_GROUP="${RUN_GROUP:-h200_3h_test_$(date +%m%d%H%M)}"
echo "Run group: $RUN_GROUP"

mkdir -p outputs/plots outputs/summaries outputs/logs

echo "Core method smoke (all methods, GSM8K)"
python -m src.run_eval --config configs/suite_smoke_full.yaml --seed 0 --run_group "$RUN_GROUP"

echo "Multiple choice checks (ARC, MMLU)"
python -m src.run_eval --config configs/arc_challenge_smoke.yaml --seed 0 --limit 30 --run_group "$RUN_GROUP"
python -m src.run_eval --config configs/mmlu_smoke.yaml --seed 0 --limit 30 --run_group "$RUN_GROUP"

echo "Code eval checks (HumanEval, MBPP)"
python -m src.run_eval --config configs/humaneval_smoke.yaml --seed 0 --limit 10 --run_group "$RUN_GROUP"
python -m src.run_eval --config configs/mbpp_smoke.yaml --seed 0 --limit 20 --run_group "$RUN_GROUP"

echo "Aggregate + diagnostics"
python scripts/aggregate_results.py --run_group "$RUN_GROUP" --bootstrap 200 --prompt_cost 0.05 --completion_cost 1.0
python scripts/plot_pareto.py --run_group "$RUN_GROUP" --grouped
python scripts/plot_pareto.py --run_group "$RUN_GROUP" --grouped --x_metric weighted
python scripts/analyze_stopping_bounds.py --run_group "$RUN_GROUP" --output_plot "outputs/plots/stopping_risk_${RUN_GROUP}.png"
python scripts/pareto_dominance.py --run_group "$RUN_GROUP" --summary_output "outputs/summaries/pareto_summary_${RUN_GROUP}.csv"
python scripts/significance_tests.py --run_group "$RUN_GROUP" --dataset gsm8k \
  --method_a anytime_sc --method_b self_consistency --a_budget 256 --a_delta 0.1 --b_budget 256
python scripts/plot_confidence_trajectory.py --run_group "$RUN_GROUP" --dataset gsm8k --method anytime_sc \
  --output "outputs/plots/confidence_${RUN_GROUP}.png"

echo "Done. Run group: $RUN_GROUP"
