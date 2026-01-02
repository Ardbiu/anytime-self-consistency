# Next Steps Report

## Status
The repository is now fully scaffolded and compliant with the "Smoke Test" requirements.

## 1. Execution Commands
To verify the pipeline from end-to-end, execute the following:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Smoke Test (Fast, CPU-friendly)
# Generates outputs/runs/gsm8k_greedy_direct_*.jsonl
make smoke

# 3. Aggregate Results
# Generates outputs/summaries/summary.csv
make aggregate

# 4. Plot Pareto Curve
# Generates outputs/plots/pareto.png
make plot
```

## 2. Expected Outputs
*   `outputs/runs/`: Will contain one JSONL file (e.g. `gsm8k_greedy_direct_20250102-....jsonl`) with 5 records.
*   `outputs/summaries/summary.csv`: Will contain a single row for the Greedy method with columns: `method`, `accuracy`, `avg_tokens`, etc.
*   `outputs/plots/pareto.png`: A scatter plot with a single "X" mark for the Greedy baseline.

## 3. Configuration Knobs
When moving to real experiments, adjust the following in `configs/gsm8k_small.yaml` or `configs/gsm8k_full.yaml`:

*   **`model_name`**: Change to `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct` for real results. The smoke test uses `Qwen2.5-0.5B-Instruct` which is fast but dumb.
*   **`limit`**: Set to `null` to run the full dataset (1319 examples for GSM8K test).
*   **`methods`**: Uncomment `self_consistency`, `best_of_n`, or `anytime_sc` blocks in the YAML.
*   **`device`**: The code auto-detects CUDA/MPS/CPU. No config change needed usually.

## 4. Limitations
*   **Smoke Model**: The 0.5B model in `configs/gsm8k_smoke.yaml` will likely get 0% accuracy on GSM8K. This is expected; it is just a pipeline test.
*   **Token Budget**: `anytime_sc` relies on `total_tokens` accumulation. Ensure your budget logic correctly accounts for prompt vs completion tokens if you care about specific cost functions. Currently it sums both.
