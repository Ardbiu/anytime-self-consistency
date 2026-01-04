# Anytime Self-Consistency via Bandits + Statistical Stopping

This repository implements "Anytime Self-Consistency", a method for efficient LLM inference that uses bandit algorithms for policy selection and statistical stopping rules to minimize token usage while maintaining high accuracy.

## Project Goal
To provide a clean, reproducible testbed for comparing:
- **Greedy Decoding**
- **Self-Consistency (Majority Vote)**
- **Best-of-N (Verifier/Scorer)**
- **Anytime Self-Consistency** (Ours)

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/anytime-sc.git
cd anytime-sc

# Create a virtual env (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

### 1. Mac / Local Development
The easiest way to verify everything works is to run the "smoke test" which uses a tiny model and runs only 5 examples.

```bash
# 1. Setup Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run Smoke Test (Greedy only, CPU-safe, <2 mins)
make smoke
# OR: python -m src.run_eval --config configs/gsm8k_smoke.yaml

# 3. Aggregate & Plot
make aggregate
make plot

# Check results
ls -l outputs/plots/pareto.png
```

### 2. Colab Pro
Use the provided runner or run commands manually:

```python
# Cell 1: Setup
!git clone https://github.com/your-username/anytime-sc.git
%cd anytime-sc
!pip install -r requirements.txt

# Cell 2: Run Experiments (e.g. Small)
!python -m src.run_eval --config configs/gsm8k_small.yaml

# Cell 3: Aggregate & Plot
!python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/summary.csv
!python scripts/plot_pareto.py --input outputs/summaries/summary.csv --output outputs/plots/pareto.png

# Cell 4: View Plot
from IPython.display import Image
Image("outputs/plots/pareto.png")
```

## What Success Looks Like
After running the pipeline, you should see:
1.  **`outputs/runs/*.jsonl`**: JSONL files containing detailed logs for each example.
2.  **`outputs/summaries/summary.csv`**: A CSV table showing accuracy, token usage, and time for each method.
3.  **`outputs/plots/pareto.png`**: A plot visualizing the trade-off between accuracy and compute (tokens).

## Running clean experiments without mixing runs

To ensure you don't plot mixed results from different experiments, the system now uses **Run IDs**.

```bash
# 1. Run Experiment (Generated files will have a unique Run ID)
make small

# 2. Aggregate & Plot ONLY the latest run
make aggregate
make plot
```

If you want to aggregate/plot a specific older run strings:
```bash
python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/summary.csv --run_id 20260102-1700
python scripts/plot_pareto.py --input outputs/summaries/summary.csv --output outputs/plots/pareto.png --run_id 20260102-1700
```

To clean up old runs but keep the latest:
```bash
make clean_keep_latest
```

## Method schema and run labels
Each config `methods` entry must include `name` plus method-specific params:
- `greedy`: optional `policy` or `prompt` (defaults to `direct`; falls back to raw question if unknown).
- `self_consistency` / `best_of_n`: require `policy`/`prompt` plus either `n_values` or `match_budgets` + `tokens_per_sample` (for budget-matched fixed-N runs).
- `self_consistency_early_stop`: requires `policy`, `n_values`, and stopping params (`stop_ratio` or `stop_count`, plus `min_samples`).
- `best_of_n_early_stop`: requires `policy`, `n_values`, and `score_threshold` (+ optional `min_samples`).
- `anytime_sc`: require `policies`, `budgets`, `deltas` (and optional `allocation` like `ucb` or `uniform`).
- `global_anytime_sc`: dataset-level global budget. Requires `policy`, `global_budget_tokens`, `init_k`, `allocation_policy`, and optional `max_samples_per_item`, `per_example_budget_tokens`, `ucb_c`, `store_allocation_steps`, `temperature`, `top_p`, `top_k`, `finalize`.

Output files are named as:
`{dataset}_{method}_{params}_{run_group}_seed{seed}_{run_id}.jsonl` (run_group omitted if not provided).

## Suite runs (multi-dataset + multi-seed)

### Local sanity suite
```bash
# Runs gsm8k + gsm_plus with 2 seeds and strict checks
bash scripts/suite_sanity.sh
```

### Global budget smoke test
```bash
bash scripts/global_smoke_test.sh
```

You can also run the suite directly:
```bash
python scripts/run_suite.py --config configs/suite_smoke.yaml --seeds 0,1 --datasets gsm8k,gsm_plus
python scripts/aggregate_results.py --latest_group --bootstrap 200
python scripts/plot_pareto.py --latest_group --grouped
python scripts/diagnose_sampling.py --latest_group
```

### GCP / paper-scale suite
```bash
python scripts/run_suite.py --config configs/suite_paper.yaml --seeds 0,1,2,3,4 --datasets gsm8k,gsm_plus,hendrycks_math
python scripts/aggregate_results.py --latest_group --bootstrap 1000
python scripts/plot_pareto.py --latest_group --grouped
python scripts/diagnose_sampling.py --latest_group
```

### Outputs to expect
- `outputs/summaries/summary_per_run.csv`: per-seed/run summary (matches one JSONL file).
- `outputs/summaries/summary_grouped.csv`: aggregated over seeds with CIs.
- `outputs/plots/pareto_grouped.png`: grouped plot with error bars.
- `outputs/summaries/summary_global_points.csv`: global budget accuracy points.
- `outputs/summaries/summary_global_curve.csv`: AUC summary for global budget curves.
- `outputs/plots/global_curve.png`: global budget curve plot.

## ICML novelty: global compute allocation
Prior work focuses on per-example budgets. We add a **global compute budget** setting where the method allocates generations across *examples* to maximize dataset accuracy. This tests adaptive allocation at the dataset level, not just stopping within a single example, and enables new policy ablations (uniform, uncertainty, VOI-lite, UCB, random, finish-one, per-example budget).
The global setting also includes deterministic baselines like uniform per-example budgets and finish-one scheduling, so gains can be attributed to adaptive allocation.

### Ablations and analysis helpers
```bash
# Ablation comparisons (allocation, fixed-N vs anytime, early-stop vs fixed-N)
python scripts/ablation_table.py --latest_group

# Significance testing (paired bootstrap)
python scripts/significance_tests.py --latest_group --dataset gsm8k --method_a anytime_sc --method_b self_consistency --a_budget 1024 --a_delta 0.05 --a_allocation ucb --b_budget 1024

# Global budget curve
python scripts/plot_global_curve.py --latest_group

# Error analysis breakdowns
python scripts/error_analysis.py --latest_group
```

## Troubleshooting
- **`ModuleNotFoundError: No module named 'src'`**: Make sure you run python from the root `anytime-sc/` directory (e.g., `python -m src.run_eval`).
- **`CUDA out of memory`**: Decrease `limit` or `batch_size` (batching not currently implemented in this simple runner, so switch to a smaller `model_name` in config).
- **`Pad token not set`**: The code attempts to fix this automatically. If it fails, try a different model family (e.g. GPT-2 doesn't have a pad token by default, Qwen does).
- **No output files**: Check if `limit` was too small or if all examples failed parsing. Check console logs.
