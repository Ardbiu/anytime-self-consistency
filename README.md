# Anytime Self-Consistency (Anytime-SC)

An experimental framework for **compute-aware LLM reasoning**. We treat inference as a **Bandits with Knapsacks (BwK)** problem and compare greedy decoding, self-consistency, best-of-N, and anytime/global allocation strategies under token budgets.

This repo is built to produce ICML-grade results: multi-seed, multi-dataset evaluation, confidence intervals, and diagnostic tooling.

## Core Ideas
- **Anytime Self-Consistency (per-example budget)**: adaptively stops sampling when the majority vote is statistically confident.
- **Global Budget Allocation (dataset-level budget)**: allocates sampling across examples to maximize accuracy under a single token budget.
- **BwK Shadow Pricing**: a primal-dual policy (Agrawal & Devanur, 2014) that penalizes expensive samples via a shadow price `lambda`.

## Methods Implemented
- `greedy`
- `self_consistency` (SC)
- `best_of_n` (BoN)
- `best_of_n_verifier` (BoN with learned verifier)
- `self_consistency_early_stop` (adaptive SC baseline)
- `best_of_n_early_stop`
- `anytime_sc` (per-example budget + bandit allocation + statistical stopping)
- `global_anytime_sc` (global budget allocation across dataset)

Allocation policies for `global_anytime_sc`:
- `uniform`, `random`, `finish_one`
- `uncertainty`, `voi` (value-of-information-lite)
- `ucb` (standard bandit)
- `per_example_budget` (deterministic baseline)
- `bwk` (primal-dual shadow pricing)

## Datasets
Supported via `src/data.py`:
- GSM8K: `openai/gsm8k`
- GSM-Plus: `qintongli/GSM-Plus`
- Hendrycks MATH: `EleutherAI/hendrycks_math`

## Installation
```bash
# Clone
git clone https://github.com/your-username/anytime-sc.git
cd anytime-sc

# Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quickstart
### Smoke Test (local)
```bash
bash scripts/smoke_test.sh
```
Artifacts:
- `outputs/summaries/smoke_summary.csv`
- `outputs/plots/smoke_pareto.png`
- `outputs/logs/smoke_<run_id>.log`

### Minimal single run
```bash
python -m src.run_eval --config configs/gsm8k_smoke.yaml
python scripts/aggregate_results.py --latest_group --bootstrap 200
python scripts/plot_pareto.py --latest_group --grouped
```

## Config Schema (methods)
Each entry in `methods` must include `name` and method-specific params:
- `greedy`: optional `policy` or `prompt` (defaults to `direct` if available).
- `self_consistency` / `best_of_n`: require `policy`/`prompt` and `n_values` OR `match_budgets` + `tokens_per_sample`.
- `best_of_n_verifier`: `policy`, `n_values`, `verifier_model_name` (optional `verifier_max_new_tokens`).
- `self_consistency_early_stop`: `policy`, `n_values`, `stop_ratio` or `stop_count`, `min_samples`.
- `best_of_n_early_stop`: `policy`, `n_values`, `score_threshold`, `min_samples`.
- `anytime_sc`: `policies`, `budgets`, `deltas`, optional `allocation`, `batch_size`, `allow_unseeded_batch`.
- `global_anytime_sc`: `policy`, `global_budget_tokens`, `init_k`, `allocation_policy` plus optional `max_samples_per_item`, `per_example_budget_tokens`, `ucb_c`, `store_allocation_steps`, `temperature`, `top_p`, `top_k`, `finalize`.

Output files are named:
```
{dataset}_{method}_{params}_{run_group}_seed{seed}_{run_id}.jsonl
```

## Running Suites
### Local sanity suite
```bash
bash scripts/suite_sanity.sh
```

### Paper-ready GSM8K hero run
```bash
python -m src.run_eval --config configs/paper_hero.yaml --seed 0 --run_group hero_gsm8k
```

### Multi-dataset hero suite
```bash
python scripts/run_suite.py --config configs/paper_hero_suite.yaml --seeds 0,1 --datasets gsm8k,gsm_plus,hendrycks_math --run_group hero_suite
```

### Model-agnostic (second model)
```bash
python -m src.run_eval --config configs/paper_model_agnostic.yaml --seed 0 --run_group model_agnostic
```

### Hard benchmark subset
```bash
python -m src.run_eval --config configs/paper_hard_math.yaml --seed 0 --run_group hard_math
```

## Aggregation & Plots
```bash
python scripts/aggregate_results.py --latest_group --bootstrap 1000
python scripts/plot_pareto.py --latest_group --grouped
python scripts/plot_pareto.py --latest_group --grouped --x_metric time
python scripts/plot_global_curve.py --latest_group
```

Outputs:
- `outputs/summaries/summary_per_run.csv`
- `outputs/summaries/summary_grouped.csv`
- `outputs/plots/pareto_grouped.png`
- `outputs/summaries/summary_global_points.csv`
- `outputs/summaries/summary_global_curve.csv`

## Diagnostics
### Sampling diversity
```bash
python scripts/diagnose_sampling.py --latest_group
```

### Stopping bound sanity check
```bash
python scripts/analyze_stopping_bounds.py --latest_group
```

### Latency benchmark
```bash
python scripts/benchmark_latency.py --model Qwen/Qwen2.5-7B-Instruct --n 10
```

### Significance tests
```bash
python scripts/significance_tests.py --latest_group \
  --dataset gsm8k --method_a anytime_sc --method_b self_consistency \
  --a_budget 1024 --a_delta 0.05 --a_allocation ucb --b_budget 1024
```

## Pseudo-validation (ucb_c)
Avoid tuning on test:
```bash
python scripts/tune_ucb_c.py --config configs/paper_hero.yaml --dataset gsm8k --limit 200 --ucb_c 0.5,1.0,2.0
python scripts/aggregate_results.py --latest_group --bootstrap 200
```
Pick the best `ucb_c` and lock it for test runs.

## BwK Shadow Pricing (Primal-Dual)
Use `allocation_policy: bwk` in `global_anytime_sc`. The shadow price is updated as:
```
lambda_{t+1} = lambda_t * exp(eta * (consumed_tokens - target_tokens))
```
and the selection index is:
```
Index = P(correct) - lambda * normalized_cost
```
Shadow price is recorded in allocation steps when `store_allocation_steps: true`.

## Notes
- `use_flash_attention` / `use_compile` appear in some configs but are not wired to model loading yet.
- Batched inference is available for SC/BoN (`batched: true`). Anytime SC has `batch_size` for batched sampling.

## Troubleshooting
- **`ModuleNotFoundError: No module named 'src'`**: run from repo root (`python -m src.run_eval`).
- **`CUDA out of memory`**: reduce `max_new_tokens` or `batch_size`, or use a smaller model.
- **No outputs**: check logs under `outputs/logs` and ensure `limit` isn’t zero.
