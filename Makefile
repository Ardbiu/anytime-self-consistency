.PHONY: smoke small full aggregate plot clean

# Install deps
install:
	pip install -r requirements.txt

# Run very fast smoke test (limit=5, small model)
smoke:
	python -m src.run_eval --config configs/gsm8k_smoke.yaml

# Run small dev run (limit=20, default model)
small:
	python -m src.run_eval --config configs/gsm8k_small.yaml

# Run full evaluation
full:
	python -m src.run_eval --config configs/gsm8k_full.yaml

# Aggregate results (latest only by default for safety in new workflows)
aggregate:
	python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/summary.csv --latest

aggregate_all:
	python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/summary.csv

# Plot results (latest only)
plot:
	python scripts/plot_pareto.py --input outputs/summaries/summary.csv --output outputs/plots/pareto.png --latest

plot_all:
	python scripts/plot_pareto.py --input outputs/summaries/summary.csv --output outputs/plots/pareto.png

# Clean outputs
clean:
	python scripts/clean_outputs.py --all

clean_keep_latest:
	python scripts/clean_outputs.py --keep_latest
