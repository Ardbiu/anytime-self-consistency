#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import pandas as pd

def load_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: Input file {path} does not exist.")
        sys.exit(1)
    df = pd.read_csv(path)
    if df.empty:
        print("Error: Empty summary CSV.")
        sys.exit(1)
    return df

def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates = (x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i]))
        if np.any(dominates):
            mask[i] = False
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Summary CSV path")
    parser.add_argument("--output", type=str, help="Optional output CSV with pareto flags")
    parser.add_argument("--summary_output", type=str, help="Optional summary CSV output")
    parser.add_argument("--latest_group", action="store_true", help="Filter to latest run_group if present")
    parser.add_argument("--run_group", type=str, help="Filter to a specific run_group")
    parser.add_argument("--latest", action="store_true", help="Filter to latest run_id if present")
    parser.add_argument("--run_id", type=str, help="Filter to a specific run_id")
    parser.add_argument("--x_metric", type=str, default="tokens", choices=["tokens", "time", "weighted"])
    args = parser.parse_args()

    default_grouped = "outputs/summaries/summary_grouped.csv"
    default_per_run = "outputs/summaries/summary_per_run.csv"
    if args.input:
        input_path = args.input
    else:
        input_path = default_grouped if os.path.exists(default_grouped) else default_per_run

    df = load_summary(input_path)

    if args.latest_group and "run_group" in df.columns:
        latest_group = df["run_group"].astype(str).max()
        df = df[df["run_group"] == latest_group]
    elif args.run_group and "run_group" in df.columns:
        df = df[df["run_group"] == args.run_group]

    if args.latest and "run_id" in df.columns:
        latest_id = df["run_id"].astype(str).max()
        df = df[df["run_id"] == latest_id]
    elif args.run_id and "run_id" in df.columns:
        df = df[df["run_id"] == args.run_id]

    if df.empty:
        print("Error: No data left after filtering.")
        sys.exit(1)

    if args.x_metric == "time":
        x_col = "mean_avg_time_s" if "mean_avg_time_s" in df.columns else "avg_time_s"
    elif args.x_metric == "weighted":
        x_col = "mean_avg_weighted_cost" if "mean_avg_weighted_cost" in df.columns else "avg_weighted_cost"
    else:
        x_col = "mean_avg_tokens" if "mean_avg_tokens" in df.columns else "avg_tokens"
    y_col = "mean_accuracy" if "mean_accuracy" in df.columns else "accuracy"

    if x_col not in df.columns or y_col not in df.columns:
        print(f"Error: Missing required columns for Pareto analysis: {x_col}, {y_col}")
        sys.exit(1)

    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        print("Error: No valid rows after dropping NaNs.")
        sys.exit(1)

    results = []
    for dataset in df["dataset"].dropna().unique().tolist():
        sub = df[df["dataset"] == dataset].copy()
        x = sub[x_col].to_numpy()
        y = sub[y_col].to_numpy()
        mask = pareto_mask(x, y)
        sub["is_pareto"] = mask

        total_pareto = int(mask.sum())
        anytime_pareto = int(sub[(sub["method"] == "anytime_sc") & sub["is_pareto"]].shape[0])
        results.append({
            "dataset": dataset,
            "pareto_points": total_pareto,
            "anytime_pareto_points": anytime_pareto,
            "anytime_pareto_share": (anytime_pareto / total_pareto) if total_pareto > 0 else np.nan,
            "anytime_total_points": int(sub[sub["method"] == "anytime_sc"].shape[0]),
        })

        df.loc[sub.index, "is_pareto"] = sub["is_pareto"]

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved Pareto flags to {args.output}")

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    if args.summary_output:
        out_dir = os.path.dirname(args.summary_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary_df.to_csv(args.summary_output, index=False)
        print(f"Saved Pareto summary to {args.summary_output}")

if __name__ == "__main__":
    main()
