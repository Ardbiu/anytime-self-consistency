# Colab Launcher Script (Copy content to a cell if needed, or run this file)
# In Colab, you would normally run these as ! commands.

import os
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"COMMAND FAILED: {cmd}")
        # Don't exit, try to continue or let user know
    else:
        print("SUCCESS")

def main():
    print("=== Anytime SC Colab Runner ===")
    
    # 1. Install defaults if needed (assuming repo clone is done)
    print("Step 1: Installing dependencies...")
    run_cmd("pip install -r requirements.txt")
    
    # 2. Run Smoke Test (Fast)
    print("\nStep 2: SMOKE TEST (Fast)")
    run_cmd("python -m src.run_eval --config configs/gsm8k_smoke.yaml")
    
    # 3. Run Small Experiment (Multi-method)
    print("\nStep 3: SMALL EXPERIMENT (Pareto Curve)")
    run_cmd("python -m src.run_eval --config configs/gsm8k_small.yaml")
    
    # 4. Aggregate
    print("\nStep 4: Aggregating Results")
    run_cmd("python scripts/aggregate_results.py --input outputs/runs --output outputs/summaries/summary.csv")
    
    # 5. Plot
    print("\nStep 5: Plotting Pareto Curve")
    run_cmd("python scripts/plot_pareto.py --input outputs/summaries/summary.csv --output outputs/plots/pareto.png")
    
    print("\n=== DONE ===")
    print("Files created:")
    run_cmd("ls -l outputs/summaries/summary.csv outputs/plots/pareto.png")

if __name__ == "__main__":
    main()
