
-# Anytime Self-Consistency via Bandits + Statistical Stopping
-
-This repository implements "Anytime Self-Consistency", a method for efficient LLM inference that uses bandit algorithms for policy selection and statistical stopping rules to minimize token usage while maintaining high accuracy.
-
-## Project Goal
-To provide a clean, reproducible testbed for comparing:
-- **Greedy Decoding**
-- **Self-Consistency (Majority Vote)**
-- **Best-of-N (Verifier/Scorer)**
-- **Anytime Self-Consistency** (Ours)
-
-## ICML-Grade Additions
-**Contextual BwK**: shadow pricing can be conditioned on prompt length or hidden-state norms (`context_config`).
-**Empirical Bernstein Bounds**: optional stopping bounds (`bound_method: empirical_bernstein`) for tighter risk control.
-**Latency Profiling**: per-example sampling/bandit/scoring timers + `latency_benefit` in outputs.
-**Resume + SIGTERM Safety**: atomic `.tmp` runs, resumable state, and graceful shutdown on clusters.
-
-## Installation
-
-```bash
-# Clone the repo
-git clone https://github.com/your-username/anytime-sc.git
-cd anytime-sc
-
-# Create a virtual env (recommended)
-python -m venv venv
-source venv/bin/activate
-
-# Install dependencies
-pip install -r requirements.txt
-```
-
-## Quickstart
-
-### 1. Mac / Local Development
-The easiest way to verify everything works is to run the "smoke test" which uses a tiny model and runs only 5 examples.
-
-```bash
-# 1. Setup Environment
-python -m venv venv
-source venv/bin/activate
-pip install -r requirements.txt
-
-# 2. Run Smoke Test (Greedy only, CPU-safe, <2 mins)
-make smoke
-# OR: python -m src.run_eval --config configs/gsm8k_smoke.yaml
-
-# 3. Aggregate & Plot
-make aggregate
-make plot
-
-# Check results
-ls -l outputs/plots/pareto.png
-```
-
-### 2. Colab Pro
-Use the provided runner or run commands manually:
-
-```python
-# Cell 1: Setup
-!git clone https://github.com/your-username/anytime-sc.git
-%cd anytime-sc
-!pip install -r requirements.txt
-
-# Cell 2: Run Experiments (e.g. Small)
-!python -m src.run_eval --config configs/gsm8k_small.yaml
-
-# Cell 3: Aggregate & Plot
-!python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/summary.csv
-!python scripts/plot_pareto.py --input outputs/summaries/summary.csv --output outputs/plots/pareto.png
-
-# Cell 4: View Plot
-from IPython.display import Image
-Image("outputs/plots/pareto.png")
-```
-
-## What Success Looks Like
-After running the pipeline, you should see:
-1.  **`outputs/runs/*.jsonl`**: JSONL files containing detailed logs for each example.
-2.  **`outputs/summaries/summary.csv`**: A CSV table showing accuracy, token usage, and time for each method.
-3.  **`outputs/plots/pareto.png`**: A plot visualizing the trade-off between accuracy and compute (tokens).

New compute-aware metrics in `outputs/summaries/summary_grouped.csv`:
- `tokens_per_correct`: Total tokens used divided by number of correct answers.
- `time_per_correct`: Total wall-clock time divided by number of correct answers.
- `accuracy_per_second`: Correct answers per second of generation time.
- `weighted_cost_per_correct`: Weighted cost per correct using prompt/completion weights.

Anytime-SC can use weighted cost budgets by setting `prompt_cost` and `completion_cost` in the method config (budgets then represent weighted cost units).

## Additional datasets (breadth)
Supported datasets beyond GSM8K/MATH:
- `arc_challenge` (ARC-Challenge multiple choice)
- `mmlu:<subject>` (e.g. `mmlu:abstract_algebra`)
- `humaneval` (code generation)
- `mbpp` (code generation)
- `gpqa` (graduate-level science multiple choice)
- `bbh` or `bbh:<task>` (Big-Bench Hard; supports few-shot prompts)

Smoke configs are provided in:
- `configs/arc_challenge_smoke.yaml`
- `configs/mmlu_smoke.yaml`
- `configs/humaneval_smoke.yaml`
- `configs/mbpp_smoke.yaml`

Note: code-task evaluation executes candidate code in a subprocess; run these in a sandboxed environment.
-
-## Running clean experiments without mixing runs
-
-To ensure you don't plot mixed results from different experiments, the system now uses **Run IDs**.
-
-```bash
-# 1. Run Experiment (Generated files will have a unique Run ID)
-make small
-
-# 2. Aggregate & Plot ONLY the latest run
-make aggregate
-make plot
-```
-
-If you want to aggregate/plot a specific older run strings:
-```bash
-python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/summary.csv --run_id 20260102-1700
-python scripts/plot_pareto.py --input outputs/summaries/summary.csv --output outputs/plots/pareto.png --run_id 20260102-1700
-```
-
-To clean up old runs but keep the latest:
-```bash
-make clean_keep_latest
-```
-
-To resume interrupted runs (cluster preemption safe):
-```bash
-python -m src.run_eval --config configs/paper_hero.yaml --resume --save_interval 10
-```
-
-## Method schema and run labels
-Each config `methods` entry must include `name` plus method-specific params:
-- `greedy`: optional `policy` or `prompt` (defaults to `direct`; falls back to raw question if unknown).
-- `self_correction`: two-pass correction baseline (optional `correction_prompt` template).
-- `speculative_decoding`: requires `draft_model_name` (optional `draft_max_new_tokens`, `draft_use_flash_attention`, `draft_use_compile`).
-- `medusa`: requires `medusa_heads` (optional `medusa_model_name`).
-- `self_consistency` / `best_of_n`: require `policy`/`prompt` plus either `n_values` or `match_budgets` + `tokens_per_sample` (for budget-matched fixed-N runs). Optional `batched` and `batched_seeded`.
-- `best_of_n_verifier`: requires `policy`/`prompt`, `n_values`, and `verifier_model_name` (optional `verifier_max_new_tokens`, `verifier_task` (yes_no or reward), `batched`, `batched_seeded`).
-- `self_consistency_early_stop`: requires `policy`, `n_values`, and stopping params (`stop_ratio` or `stop_count`, plus `min_samples`).
-- `best_of_n_early_stop`: requires `policy`, `n_values`, and `score_threshold` (+ optional `min_samples`).
-- `anytime_sc`: require `policies`, `budgets`, `deltas` (optional `allocation` like `ucb`, `ucb_window`, `ucb_discount`, `uniform`, `bwk`, `contextual_bwk`; plus `batch_size`, `allow_unseeded_batch`, `ucb_window`, `ucb_discount`, `prompt_cost`, `completion_cost`, `bound_method`, `context_config`, `context_policy`, `safety_valve`).
-- `oracle_stopping`: require `policies`, `budgets` (optional `allocation`, `batch_size`, `allow_unseeded_batch`, `ucb_window`, `ucb_discount`, `prompt_cost`, `completion_cost`).
-- `global_anytime_sc`: dataset-level global budget. Requires `policy`, `global_budget_tokens`, `init_k`, `allocation_policy`, and optional `max_samples_per_item`, `per_example_budget_tokens`, `ucb_c`, `store_allocation_steps`, `temperature`, `top_p`, `top_k`, `finalize`, `context_config`.
-Available prompts include `direct`, `cot`, `cot_long` (verbose CoT for compute-equivalent greedy), and `decompose`.
-
-Output files are named as:
-`{dataset}_{method}_{params}_{run_group}_seed{seed}_{run_id}.jsonl` (run_group omitted if not provided).
-
-## Suite runs (multi-dataset + multi-seed)
-
-### Local sanity suite
-```bash
-# Runs gsm8k + gsm_plus with 2 seeds and strict checks
-bash scripts/suite_sanity.sh
-```
-
-### Hero configs (paper-ready)
-```bash
-# Full GSM8K with strong baselines (greedy + SC + Adaptive SC + Anytime + Global)
-python -m src.run_eval --config configs/paper_hero.yaml --seed 0 --run_group hero_gsm8k
-
-# GSM8K full + GSM-Plus (limited) with the same hero methods
-python scripts/run_suite.py --config configs/paper_hero_suite.yaml --seeds 0,1 --datasets gsm8k,gsm_plus --run_group hero_suite
-
-# 5-seed hero suite (paper-grade stability)
-bash scripts/run_hero_5seed.sh
-```
-
-### Model-agnostic and hard-benchmark runs
-```bash
-# Second model family (Mistral 7B) on a 250-sample GSM8K subset
-python -m src.run_eval --config configs/paper_model_agnostic.yaml --seed 0 --run_group model_agnostic
-
-# Hard benchmark (Hendrycks MATH) subset
-python -m src.run_eval --config configs/paper_hard_math.yaml --seed 0 --run_group hard_math
-```
-
-### Global budget smoke test
-```bash
-bash scripts/global_smoke_test.sh
-```
-
-You can also run the suite directly:
-```bash
-python scripts/run_suite.py --config configs/suite_smoke.yaml --seeds 0,1 --datasets gsm8k,gsm_plus
-# Optional checkpointing to stop weak methods early (saves tokens)
-python scripts/run_suite.py --config configs/suite_smoke.yaml --seeds 0,1 --datasets gsm8k,gsm_plus --checkpoint_examples 50 --checkpoint_degradation 0.2
-python scripts/aggregate_results.py --latest_group --bootstrap 200
-python scripts/plot_pareto.py --latest_group --grouped
-python scripts/diagnose_sampling.py --latest_group
-```
-
-### GCP / paper-scale suite
-```bash
-python scripts/run_suite.py --config configs/suite_paper.yaml --seeds 0,1,2,3,4 --datasets gsm8k,gsm_plus,hendrycks_math
-python scripts/aggregate_results.py --latest_group --bootstrap 1000
-python scripts/plot_pareto.py --latest_group --grouped
-python scripts/diagnose_sampling.py --latest_group
-```
-
-### Fleet manager (cluster autopilot)
-```bash
-bash fleet_manager.sh
-```
-
-## Pseudo-validation for ucb_c
-To avoid tuning on test, sweep `ucb_c` on a small training subset (first 200 examples):
-```bash
-python scripts/tune_ucb_c.py --config configs/paper_hero.yaml --dataset gsm8k --limit 200 --ucb_c 0.5,1.0,2.0
-python scripts/aggregate_results.py --latest_group --bootstrap 200
-```
-Pick the best `ucb_c` from the grouped summary and use it in your final test run.
-
-### Outputs to expect
-- `outputs/summaries/summary_per_run.csv`: per-seed/run summary (matches one JSONL file).
-- `outputs/summaries/summary_grouped.csv`: aggregated over seeds with CIs.
-- `outputs/plots/pareto_grouped.png`: grouped plot with error bars.
-- `outputs/summaries/summary_global_points.csv`: global budget accuracy points.
-- `outputs/summaries/summary_global_curve.csv`: AUC summary for global budget curves.
-- `outputs/plots/global_curve.png`: global budget curve plot.
-
-## ICML novelty: global compute allocation
-Prior work focuses on per-example budgets. We add a **global compute budget** setting where the method allocates generations across *examples* to maximize dataset accuracy. This tests adaptive allocation at the dataset level, not just stopping within a single example, and enables new policy ablations (uniform, uncertainty, VOI-lite, UCB, random, finish-one, per-example budget).
-The global setting also includes deterministic baselines like uniform per-example budgets and finish-one scheduling, so gains can be attributed to adaptive allocation.
-
-### Ablations and analysis helpers
-```bash
-# Ablation comparisons (allocation, fixed-N vs anytime, early-stop vs fixed-N)
-python scripts/ablation_table.py --latest_group
-
-# Significance testing (paired bootstrap)
-python scripts/significance_tests.py --latest_group --dataset gsm8k --method_a anytime_sc --method_b self_consistency --a_budget 1024 --a_delta 0.05 --a_allocation ucb --b_budget 1024
-
-# Global budget curve
-python scripts/plot_global_curve.py --latest_group
-
-# Accuracy vs wall-clock time (grouped)
-python scripts/plot_pareto.py --latest_group --grouped --x_metric time
-
-# Weighted cost (prompt vs completion) ablation
-python scripts/aggregate_results.py --latest_group --prompt_cost 0.1 --completion_cost 1.0
-python scripts/plot_pareto.py --latest_group --grouped --x_metric weighted
-
-# Pareto dominance report (checks frontier coverage)
-python scripts/pareto_dominance.py --latest_group --summary_output outputs/summaries/pareto_summary.csv
-
-# Confidence trajectory plots (entropy vs samples)
-python scripts/plot_confidence_trajectory.py --latest_group --dataset gsm8k --method anytime_sc
-
-# Error analysis breakdowns
-python scripts/error_analysis.py --latest_group
-```
-
-### Theoretical sanity checks
-Validate that the statistical stopping rule honors the delta (risk) parameter:
-```bash
-python scripts/analyze_stopping_bounds.py --latest_group
-```
-This compares empirical error rates against Hoeffding bounds and outputs a pass/fail summary.
-You can also emit a delta-risk plot:
-```bash
-python scripts/analyze_stopping_bounds.py --latest_group --output_plot outputs/plots/stopping_risk.png
-```
-
-### Latency benchmarks
-Compare serial vs batched inference to demonstrate wall-clock savings:
-```bash
-python scripts/benchmark_latency.py --model Qwen/Qwen2.5-7B-Instruct --n 10
-```
-Batched inference is available in `run_self_consistency()` and `run_best_of_n()` via the `batched=True` parameter.
-
-## Troubleshooting
-- **`ModuleNotFoundError: No module named 'src'`**: Make sure you run python from the root `anytime-sc/` directory (e.g., `python -m src.run_eval`).
-- **`CUDA out of memory`**: Decrease `limit` or `max_new_tokens`; if you enable batched anytime sampling, reduce `batch_size`.
-- **`Pad token not set`**: The code attempts to fix this automatically. If it fails, try a different model family (e.g. GPT-2 doesn't have a pad token by default, Qwen does).
-- **No output files**: Check if `limit` was too small or if all examples failed parsing. Check console logs.
+# Anytime Self-Consistency (Anytime-SC)
+
+An experimental framework for **compute-aware LLM reasoning**. We treat inference as a **Bandits with Knapsacks (BwK)** problem and compare greedy decoding, self-consistency, best-of-N, and anytime/global allocation strategies under token budgets.
+
+This repo is built to produce ICML-grade results: multi-seed, multi-dataset evaluation, confidence intervals, and diagnostic tooling.
+
+## Core Ideas
+- **Anytime Self-Consistency (per-example budget)**: adaptively stops sampling when the majority vote is statistically confident.
+- **Global Budget Allocation (dataset-level budget)**: allocates sampling across examples to maximize accuracy under a single token budget.
+- **BwK Shadow Pricing**: a primal-dual policy (Agrawal & Devanur, 2014) that penalizes expensive samples via a shadow price `lambda`.
+
+## Methods Implemented
+- `greedy`
+- `self_consistency` (SC)
+- `best_of_n` (BoN)
+- `best_of_n_verifier` (BoN with learned verifier)
+- `self_consistency_early_stop` (adaptive SC baseline)
+- `best_of_n_early_stop`
+- `anytime_sc` (per-example budget + bandit allocation + statistical stopping)
+- `global_anytime_sc` (global budget allocation across dataset)
+
+Allocation policies for `global_anytime_sc`:
+- `uniform`, `random`, `finish_one`
+- `uncertainty`, `voi` (value-of-information-lite)
+- `voc_anytime` (value-of-computation heuristic)
+- `ucb` (standard bandit)
+- `per_example_budget` (deterministic baseline)
+- `bwk` (primal-dual shadow pricing)
+
+## Datasets
+Supported via `src/data.py`:
+- GSM8K: `openai/gsm8k`
+- GSM-Plus: `qintongli/GSM-Plus`
+- Hendrycks MATH: `EleutherAI/hendrycks_math`
+
+## Installation
+```bash
+# Clone
+git clone https://github.com/your-username/anytime-sc.git
+cd anytime-sc
+
+# Environment
+python -m venv venv
+source venv/bin/activate
+pip install -r requirements.txt
+```
+
+## Quickstart
+### Smoke Test (local)
+```bash
+bash scripts/smoke_test.sh
+```
+Artifacts:
+- `outputs/summaries/smoke_summary.csv`
+- `outputs/plots/smoke_pareto.png`
+- `outputs/logs/smoke_<run_id>.log`
+
+### Minimal single run
+```bash
+python -m src.run_eval --config configs/gsm8k_smoke.yaml
+python scripts/aggregate_results.py --latest_group --bootstrap 200
+python scripts/plot_pareto.py --latest_group --grouped
+```
+
+## Config Schema (methods)
+Each entry in `methods` must include `name` and method-specific params:
+- `greedy`: optional `policy` or `prompt` (defaults to `direct` if available).
+- `self_consistency` / `best_of_n`: require `policy`/`prompt` and `n_values` OR `match_budgets` + `tokens_per_sample` (optional `batched`, `batched_seeded`).
+- `best_of_n_verifier`: `policy`, `n_values`, `verifier_model_name` (optional `verifier_max_new_tokens`, `verifier_task` (yes_no or reward), `batched`, `batched_seeded`).
+- `self_consistency_early_stop`: `policy`, `n_values`, `stop_ratio` or `stop_count`, `min_samples`.
+- `best_of_n_early_stop`: `policy`, `n_values`, `score_threshold`, `min_samples`.
+- `anytime_sc`: `policies`, `budgets`, `deltas`, optional `allocation`, `batch_size`, `allow_unseeded_batch`.
+- `global_anytime_sc`: `policy`, `global_budget_tokens`, `init_k`, `allocation_policy` plus optional `max_samples_per_item`, `per_example_budget_tokens`, `ucb_c`, `store_allocation_steps`, `temperature`, `top_p`, `top_k`, `finalize`.
+
+Output files are named:
+```
+{dataset}_{method}_{params}_{run_group}_seed{seed}_{run_id}.jsonl
+```
+
+## Running Suites
+### Local sanity suite
+```bash
+bash scripts/suite_sanity.sh
+```
+
+### Paper-ready GSM8K hero run
+```bash
+python -m src.run_eval --config configs/paper_hero.yaml --seed 0 --run_group hero_gsm8k
+```
+
+### Multi-dataset hero suite
+```bash
+python scripts/run_suite.py --config configs/paper_hero_suite.yaml --seeds 0,1 --datasets gsm8k,gsm_plus,hendrycks_math --run_group hero_suite
+```
+
+### Model-agnostic (second model)
+```bash
+python -m src.run_eval --config configs/paper_model_agnostic.yaml --seed 0 --run_group model_agnostic
+```
+
+### Hard benchmark subset
+```bash
+python -m src.run_eval --config configs/paper_hard_math.yaml --seed 0 --run_group hard_math
+```
+
+## Aggregation & Plots
+```bash
+python scripts/aggregate_results.py --latest_group --bootstrap 1000
+python scripts/plot_pareto.py --latest_group --grouped
+python scripts/plot_pareto.py --latest_group --grouped --x_metric time
+python scripts/plot_global_curve.py --latest_group
+```
+
+Outputs:
+- `outputs/summaries/summary_per_run.csv`
+- `outputs/summaries/summary_grouped.csv`
+- `outputs/plots/pareto_grouped.png`
+- `outputs/summaries/summary_global_points.csv`
+- `outputs/summaries/summary_global_curve.csv`
+
+## Diagnostics
+### Sampling diversity
+```bash
+python scripts/diagnose_sampling.py --latest_group
+```
+
+### Stopping bound sanity check
+```bash
+python scripts/analyze_stopping_bounds.py --latest_group
+```
+
+### Latency benchmark
+```bash
+python scripts/benchmark_latency.py --model Qwen/Qwen2.5-7B-Instruct --n 10
+```
+
+### Significance tests
+```bash
+python scripts/significance_tests.py --latest_group \\
+  --dataset gsm8k --method_a anytime_sc --method_b self_consistency \\
+  --a_budget 1024 --a_delta 0.05 --a_allocation ucb --b_budget 1024
+```
+
+## Pseudo-validation (ucb_c)
+Avoid tuning on test:
+```bash
+python scripts/tune_ucb_c.py --config configs/paper_hero.yaml --dataset gsm8k --limit 200 --ucb_c 0.5,1.0,2.0
+python scripts/aggregate_results.py --latest_group --bootstrap 200
+```
+Pick the best `ucb_c` and lock it for test runs.
+
+## BwK Shadow Pricing (Primal-Dual)
+Use `allocation_policy: bwk` in `global_anytime_sc`. The shadow price is updated as:
+```
+lambda_{t+1} = lambda_t * exp(eta * (consumed_tokens - target_tokens))
+```
+and the selection index is:
+```
+Index = P(correct) - lambda * normalized_cost
+```
+Shadow price is recorded in allocation steps when `store_allocation_steps: true`.
+
+## Notes
+- `use_flash_attention` / `use_compile` are wired into model loading; verifier models can override with `verifier_use_flash_attention` / `verifier_use_compile`.
+- Batched inference is available for SC/BoN (`batched: true`). Set `batched_seeded: true` for deterministic but slower sampling. Anytime SC uses `batch_size` and `allow_unseeded_batch`.
+
+## Troubleshooting
+- **`ModuleNotFoundError: No module named 'src'`**: run from repo root (`python -m src.run_eval`).
+- **`CUDA out of memory`**: reduce `max_new_tokens` or `batch_size`, or use a smaller model.
+- **No outputs**: check logs under `outputs/logs` and ensure `limit` isn’t zero.
