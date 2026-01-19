#!/usr/bin/env python
"""
Theoretical Sanity Check: Analyze Stopping Bounds

This script validates that the statistical stopping rule in anytime_sc
actually honors the delta (risk) parameter by comparing:
- Empirical error rate: Fraction of stopped examples where the final answer was wrong
- Theoretical bound: The delta parameter from the config

If the Hoeffding-based stopping rule is working correctly, we expect:
    empirical_error_rate <= delta (with high probability)
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def read_first_record(path: str) -> dict:
    """Read the first JSONL record to get metadata."""
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except Exception:
        return {}
    return {}


def get_latest_run_group(files: List[str]) -> Optional[str]:
    """Find the most recent run_group from a list of JSONL files."""
    run_groups = set()
    for f in files:
        rec = read_first_record(f)
        rg = rec.get("run_group")
        if rg:
            run_groups.add(rg)
    if not run_groups:
        return None
    return sorted(run_groups)[-1]


def load_anytime_sc_records(
    files: List[str], run_group: Optional[str] = None
) -> List[dict]:
    """Load all anytime_sc records from JSONL files."""
    records = []
    for f in files:
        try:
            with open(f, "r") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # Filter by method
                    if rec.get("method") != "anytime_sc":
                        continue
                    # Filter by run_group if specified
                    if run_group and rec.get("run_group") != run_group:
                        continue
                    records.append(rec)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}", file=sys.stderr)
    return records


def compute_hoeffding_bound(t: int, delta: float) -> float:
    """
    Compute the Hoeffding-based stopping threshold.
    
    The stopping rule is: (c1 - c2) >= sqrt(2 * t * log(1/delta))
    
    This is derived from Hoeffding's inequality for the difference
    between the empirical proportion and true proportion.
    """
    if t <= 0 or delta <= 0:
        return 0.0
    return math.sqrt(2 * t * math.log(1 / delta))


def analyze_single_record(rec: dict) -> dict:
    """Analyze stopping behavior for a single anytime_sc record."""
    steps = rec.get("steps", [])
    delta = rec.get("delta", 0.05)
    is_correct = rec.get("is_correct", False)
    
    if not steps:
        return None
    
    # Find if/when we stopped
    stop_step = None
    for step in steps:
        if step.get("stop", False):
            stop_step = step
            break
    
    # If no explicit stop, we hit budget limit
    final_step = steps[-1]
    stopped_early = stop_step is not None
    
    # Get final margin and threshold
    if stopped_early:
        t = stop_step["t"]
        margin = stop_step["margin"]
        threshold = stop_step["threshold"]
    else:
        t = final_step["t"]
        margin = final_step["margin"]
        threshold = final_step["threshold"]
    
    # Compute theoretical bound
    theoretical_threshold = compute_hoeffding_bound(t, delta)
    
    # Check if the bound was satisfied at stopping
    bound_satisfied = margin >= threshold
    
    return {
        "example_id": rec.get("example_id"),
        "delta": delta,
        "num_samples": t,
        "stopped_early": stopped_early,
        "margin_at_stop": margin,
        "threshold_at_stop": threshold,
        "theoretical_threshold": theoretical_threshold,
        "bound_satisfied": bound_satisfied,
        "is_correct": is_correct,
        "budget_tokens": rec.get("budget_tokens"),
        "allocation": rec.get("allocation"),
    }


def aggregate_by_delta(analyses: List[dict]) -> pd.DataFrame:
    """Aggregate analyses by delta value."""
    grouped = defaultdict(list)
    for a in analyses:
        if a is None:
            continue
        grouped[a["delta"]].append(a)
    
    rows = []
    for delta, items in sorted(grouped.items()):
        n_total = len(items)
        n_stopped_early = sum(1 for i in items if i["stopped_early"])
        n_correct = sum(1 for i in items if i["is_correct"])
        n_wrong = n_total - n_correct
        
        # Among those who stopped early, how many were wrong?
        early_stopped = [i for i in items if i["stopped_early"]]
        n_early_wrong = sum(1 for i in early_stopped if not i["is_correct"])
        
        # Empirical error rate among early-stopped
        if len(early_stopped) > 0:
            empirical_error_rate = n_early_wrong / len(early_stopped)
        else:
            empirical_error_rate = 0.0
        
        # Average number of samples
        avg_samples = np.mean([i["num_samples"] for i in items])
        
        # Check if bound is honored
        bound_honored = empirical_error_rate <= delta
        
        rows.append({
            "delta": delta,
            "n_examples": n_total,
            "n_stopped_early": n_stopped_early,
            "early_stop_rate": n_stopped_early / n_total if n_total > 0 else 0,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "accuracy": n_correct / n_total if n_total > 0 else 0,
            "n_early_wrong": n_early_wrong,
            "empirical_error_rate": empirical_error_rate,
            "theoretical_bound": delta,
            "bound_honored": bound_honored,
            "avg_samples": avg_samples,
        })
    
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("THEORETICAL SANITY CHECK: Stopping Bound Analysis")
    print("=" * 70)
    
    for _, row in df.iterrows():
        status = "✓ PASS" if row["bound_honored"] else "✗ FAIL"
        print(f"\nDelta = {row['delta']:.3f}")
        print(f"  Examples: {row['n_examples']}")
        print(f"  Early-stopped: {row['n_stopped_early']} ({row['early_stop_rate']*100:.1f}%)")
        print(f"  Accuracy: {row['accuracy']*100:.1f}%")
        print(f"  Empirical Error Rate (early-stopped): {row['empirical_error_rate']*100:.2f}%")
        print(f"  Theoretical Bound (delta): {row['theoretical_bound']*100:.2f}%")
        print(f"  Bound Honored: {status}")
        print(f"  Avg Samples at Stop: {row['avg_samples']:.1f}")
    
    print("\n" + "=" * 70)
    all_honored = df["bound_honored"].all() if len(df) > 0 else True
    if all_honored:
        print("OVERALL: ✓ All bounds honored. Statistical stopping is valid.")
    else:
        print("OVERALL: ✗ Some bounds violated. Investigate further.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stopping bounds for anytime_sc method"
    )
    parser.add_argument(
        "--dir",
        default="outputs/runs",
        help="Directory containing JSONL run files"
    )
    parser.add_argument(
        "--run_group",
        type=str,
        help="Specific run_group to analyze"
    )
    parser.add_argument(
        "--latest_group",
        action="store_true",
        help="Analyze the latest run_group"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/summaries/stopping_bounds_analysis.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        help="Optional output PNG path for delta-risk plot"
    )
    args = parser.parse_args()
    
    # Find JSONL files
    pattern = os.path.join(args.dir, "*.jsonl")
    files = glob.glob(pattern)
    if not files:
        print(f"Error: No JSONL files found in {args.dir}", file=sys.stderr)
        sys.exit(1)
    
    # Determine run_group
    run_group = args.run_group
    if args.latest_group:
        run_group = get_latest_run_group(files)
        if not run_group:
            print("Error: Could not determine latest run_group", file=sys.stderr)
            sys.exit(1)
        print(f"Using latest run_group: {run_group}")
    
    # Load records
    records = load_anytime_sc_records(files, run_group)
    if not records:
        print("Error: No anytime_sc records found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(records)} anytime_sc records")
    
    # Analyze each record
    analyses = [analyze_single_record(rec) for rec in records]
    analyses = [a for a in analyses if a is not None]
    
    if not analyses:
        print("Error: No valid analyses produced", file=sys.stderr)
        sys.exit(1)
    
    # Aggregate by delta
    summary_df = aggregate_by_delta(analyses)
    
    # Print summary
    print_summary(summary_df)
    
    # Save to CSV
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print(f"Saved detailed analysis to {args.output}")
    
    # Also save per-example analysis
    detail_path = args.output.replace(".csv", "_detail.csv")
    pd.DataFrame(analyses).to_csv(detail_path, index=False)
    print(f"Saved per-example details to {detail_path}")

    if args.output_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Warning: matplotlib not available for plotting: {e}", file=sys.stderr)
        else:
            plot_df = summary_df.dropna(subset=["delta", "empirical_error_rate"])
            if plot_df.empty:
                print("Warning: No data available for plotting.", file=sys.stderr)
            else:
                plt.figure(figsize=(6, 4))
                plt.plot(plot_df["delta"], plot_df["empirical_error_rate"], marker="o", label="Empirical error")
                plt.plot(plot_df["delta"], plot_df["delta"], linestyle="--", label="y = delta")
                plt.xlabel("Delta")
                plt.ylabel("Empirical Error Rate")
                plt.title("Stopping Risk vs Delta")
                plt.legend()
                out_dir = os.path.dirname(args.output_plot)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(args.output_plot, dpi=150)
                print(f"Saved delta-risk plot to {args.output_plot}")


if __name__ == "__main__":
    main()
