import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input CSV with global points")
    parser.add_argument("--output", type=str, help="Output plot PNG")
    parser.add_argument("--run_group", type=str, help="Filter to run_group")
    parser.add_argument("--latest_group", action="store_true", help="Use latest run_group")
    args = parser.parse_args()

    default_input = "outputs/summaries/summary_global_points.csv"
    input_path = args.input or default_input
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if df.empty:
        print("Error: Empty global points CSV.")
        sys.exit(1)

    if args.latest_group and "run_group" in df.columns:
        latest = df["run_group"].astype(str).max()
        df = df[df["run_group"] == latest]
        print(f"Filtering to latest run_group: {latest}")
    elif args.run_group and "run_group" in df.columns:
        df = df[df["run_group"] == args.run_group]
        print(f"Filtering to run_group: {args.run_group}")

    if df.empty:
        print("Error: No data left after filtering.")
        sys.exit(1)

    if not args.output:
        args.output = "outputs/plots/global_curve.png"

    datasets = df["dataset"].dropna().unique().tolist()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 6 * len(datasets)), squeeze=False)

    def error_arrays(sub_df, col, low_col, high_col):
        lows = sub_df[low_col] if low_col in sub_df.columns else pd.Series([np.nan] * len(sub_df))
        highs = sub_df[high_col] if high_col in sub_df.columns else pd.Series([np.nan] * len(sub_df))
        vals = sub_df[col]
        lower = vals - lows
        upper = highs - vals
        lower = lower.where(np.isfinite(lower), 0.0).to_numpy()
        upper = upper.where(np.isfinite(upper), 0.0).to_numpy()
        return [lower, upper]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx][0]
        sub_df = df[df["dataset"] == dataset]
        group_cols = ["allocation", "policy", "init_k", "max_samples_per_item"]
        for keys, gdf in sub_df.groupby(group_cols, dropna=False):
            gdf = gdf.dropna(subset=["mean_total_tokens_sum", "mean_accuracy"]).sort_values("mean_total_tokens_sum")
            if gdf.empty:
                continue
            alloc, policy, init_k, max_k = keys
            max_k_lbl = "none" if pd.isna(max_k) else int(max_k)
            label = f"{alloc}|{policy}|init{int(init_k)}|max{max_k_lbl}"
            xerr = error_arrays(gdf, "mean_total_tokens_sum", "total_tokens_sum_ci_low", "total_tokens_sum_ci_high")
            yerr = error_arrays(gdf, "mean_accuracy", "accuracy_ci_low", "accuracy_ci_high")
            ax.errorbar(
                gdf["mean_total_tokens_sum"],
                gdf["mean_accuracy"],
                xerr=xerr,
                yerr=yerr,
                marker="o",
                linestyle="-",
                label=label,
            )

        ax.set_title(f"Global Budget Curve ({dataset})")
        ax.set_xlabel("Total Tokens (Global)")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
