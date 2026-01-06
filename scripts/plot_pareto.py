import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input summary CSV file")
    parser.add_argument("--output", type=str, help="Output plot PNG file")
    parser.add_argument("--run_id", type=str, help="Specific run_id to plot")
    parser.add_argument("--latest", action="store_true", help="Plot only the latest run_id found in summary")
    parser.add_argument("--run_group", type=str, help="Specific run_group to plot")
    parser.add_argument("--latest_group", action="store_true", help="Plot only the latest run_group found in summary")
    parser.add_argument("--grouped", action="store_true", help="Plot grouped summary with CIs")
    parser.add_argument("--x_metric", type=str, default="tokens", choices=["tokens", "time"], help="X-axis metric: tokens or time")
    args = parser.parse_args()

    default_grouped = "outputs/summaries/summary_grouped.csv"
    default_per_run = "outputs/summaries/summary_per_run.csv"
    if args.input:
        input_path = args.input
    else:
        if args.grouped and os.path.exists(default_grouped):
            input_path = default_grouped
        elif os.path.exists(default_grouped):
            input_path = default_grouped
            args.grouped = True
        else:
            input_path = default_per_run

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if df.empty:
        print("Error: Empty summary CSV.")
        sys.exit(1)

    if args.grouped and "run_group" not in df.columns:
        args.grouped = False

    if args.latest_group:
        if "run_group" not in df.columns:
            print("Warning: 'run_group' column not found in summary. Cannot filter by latest group.")
        else:
            latest_group = df["run_group"].astype(str).max()
            print(f"Filtering to latest run_group: {latest_group}")
            df = df[df["run_group"] == latest_group]
    elif args.run_group:
        if "run_group" not in df.columns:
            print("Warning: 'run_group' column not found in summary.")
        else:
            print(f"Filtering to run_group: {args.run_group}")
            df = df[df["run_group"] == args.run_group]
    elif args.latest:
        if "run_id" not in df.columns:
            print("Warning: 'run_id' column not found in summary. Cannot filter by latest.")
        else:
            latest_id = df["run_id"].astype(str).max()
            print(f"Filtering to latest run_id: {latest_id}")
            df = df[df["run_id"] == latest_id]
    elif args.run_id:
        if "run_id" not in df.columns:
            print("Warning: 'run_id' column not found in summary.")
        else:
            print(f"Filtering to run_id: {args.run_id}")
            df = df[df["run_id"] == args.run_id]

    if df.empty:
        print("Error: No data left after filtering.")
        sys.exit(1)

    if not args.output:
        if args.grouped:
            args.output = "outputs/plots/pareto_grouped.png"
        else:
            args.output = "outputs/plots/pareto.png"

    if args.x_metric == "time":
        x_col = "mean_avg_time_s" if args.grouped else "avg_time_s"
        x_low_col = "time_ci_low" if args.grouped else None
        x_high_col = "time_ci_high" if args.grouped else None
        x_label = "Average Time per Example (s)"
    else:
        x_col = "mean_avg_tokens" if args.grouped else "avg_tokens"
        x_low_col = "tokens_ci_low" if args.grouped else "avg_tokens_ci_low"
        x_high_col = "tokens_ci_high" if args.grouped else "avg_tokens_ci_high"
        x_label = "Average Total Tokens per Example"

    y_col = "mean_accuracy" if args.grouped else "accuracy"
    y_low_col = "accuracy_ci_low"
    y_high_col = "accuracy_ci_high"

    if x_col not in df.columns:
        print(f"Error: X-axis column '{x_col}' not found in summary. Did you run aggregate_results with time metrics?")
        sys.exit(1)

    datasets = df["dataset"].dropna().unique().tolist() if "dataset" in df.columns else ["all"]
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
        sub_df = df if dataset == "all" else df[df["dataset"] == dataset]

        greedy = sub_df[sub_df["method"] == "greedy"].dropna(subset=[x_col, y_col])
        if not greedy.empty:
            xerr = error_arrays(greedy, x_col, x_low_col, x_high_col)
            yerr = error_arrays(greedy, y_col, y_low_col, y_high_col)
            ax.errorbar(greedy[x_col], greedy[y_col], xerr=xerr, yerr=yerr, label="Greedy", marker="x", linestyle="None", color="black", zorder=5)

        sc = sub_df[sub_df["method"] == "self_consistency"].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if not sc.empty:
            xerr = error_arrays(sc, x_col, x_low_col, x_high_col)
            yerr = error_arrays(sc, y_col, y_low_col, y_high_col)
            ax.errorbar(sc[x_col], sc[y_col], xerr=xerr, yerr=yerr, label="Self-Consistency", marker="o", linestyle="-")
            for _, row in sc.iterrows():
                lbl = f"n={int(row['n'])}"
                if "budget" in row and not pd.isna(row["budget"]):
                    lbl += f", b={int(row['budget'])}"
                ax.annotate(lbl, (row[x_col], row[y_col]), xytext=(0, 5), textcoords='offset points', fontsize=8)

        sc_es = sub_df[sub_df["method"] == "self_consistency_early_stop"].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if not sc_es.empty:
            xerr = error_arrays(sc_es, x_col, x_low_col, x_high_col)
            yerr = error_arrays(sc_es, y_col, y_low_col, y_high_col)
            ax.errorbar(sc_es[x_col], sc_es[y_col], xerr=xerr, yerr=yerr, label="SC Early-Stop", marker="D", linestyle=":")
            for _, row in sc_es.iterrows():
                ax.annotate(f"n={int(row['n'])}", (row[x_col], row[y_col]), xytext=(0, 5), textcoords='offset points', fontsize=8)

        bon = sub_df[sub_df["method"] == "best_of_n"].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if not bon.empty:
            xerr = error_arrays(bon, x_col, x_low_col, x_high_col)
            yerr = error_arrays(bon, y_col, y_low_col, y_high_col)
            ax.errorbar(bon[x_col], bon[y_col], xerr=xerr, yerr=yerr, label="Best-of-N", marker="s", linestyle="--")
            for _, row in bon.iterrows():
                lbl = f"n={int(row['n'])}"
                if "budget" in row and not pd.isna(row["budget"]):
                    lbl += f", b={int(row['budget'])}"
                ax.annotate(lbl, (row[x_col], row[y_col]), xytext=(0, -10), textcoords='offset points', fontsize=8)

        bon_ver = sub_df[sub_df["method"] == "best_of_n_verifier"].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if not bon_ver.empty:
            if "verifier_model_name" in bon_ver.columns:
                for verifier_name, vdf in bon_ver.groupby("verifier_model_name", dropna=False):
                    vdf = vdf.sort_values(x_col)
                    short_name = None if pd.isna(verifier_name) else str(verifier_name).split("/")[-1]
                    label = "BoN Verifier" if not short_name else f"BoN Verifier ({short_name})"
                    xerr = error_arrays(vdf, x_col, x_low_col, x_high_col)
                    yerr = error_arrays(vdf, y_col, y_low_col, y_high_col)
                    ax.errorbar(vdf[x_col], vdf[y_col], xerr=xerr, yerr=yerr, label=label, marker="X", linestyle="-.")
                    for _, row in vdf.iterrows():
                        lbl = f"n={int(row['n'])}"
                        if "budget" in row and not pd.isna(row["budget"]):
                            lbl += f", b={int(row['budget'])}"
                        ax.annotate(lbl, (row[x_col], row[y_col]), xytext=(0, -12), textcoords='offset points', fontsize=8)
            else:
                xerr = error_arrays(bon_ver, x_col, x_low_col, x_high_col)
                yerr = error_arrays(bon_ver, y_col, y_low_col, y_high_col)
                ax.errorbar(bon_ver[x_col], bon_ver[y_col], xerr=xerr, yerr=yerr, label="BoN Verifier", marker="X", linestyle="-.")

        bon_es = sub_df[sub_df["method"] == "best_of_n_early_stop"].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if not bon_es.empty:
            xerr = error_arrays(bon_es, x_col, x_low_col, x_high_col)
            yerr = error_arrays(bon_es, y_col, y_low_col, y_high_col)
            ax.errorbar(bon_es[x_col], bon_es[y_col], xerr=xerr, yerr=yerr, label="BoN Early-Stop", marker="P", linestyle=":")
            for _, row in bon_es.iterrows():
                ax.annotate(f"n={int(row['n'])}", (row[x_col], row[y_col]), xytext=(0, -10), textcoords='offset points', fontsize=8)

        anytime = sub_df[sub_df["method"] == "anytime_sc"].dropna(subset=[x_col, y_col])
        if not anytime.empty:
            if "allocation" in anytime.columns:
                allocs = anytime["allocation"].dropna().unique()
                for alloc in allocs:
                    alloc_df = anytime[anytime["allocation"] == alloc].sort_values(x_col)
                    xerr = error_arrays(alloc_df, x_col, x_low_col, x_high_col)
                    yerr = error_arrays(alloc_df, y_col, y_low_col, y_high_col)
                    label_txt = f"Anytime ({alloc})"
                    ax.errorbar(alloc_df[x_col], alloc_df[y_col], xerr=xerr, yerr=yerr, marker="^", linestyle="-", label=label_txt)
                    for _, row in alloc_df.iterrows():
                        lbl = f"b={int(row['budget'])}"
                        if "delta" in row and not pd.isna(row["delta"]):
                            lbl += f", d={row['delta']}"
                        ax.annotate(lbl, (row[x_col], row[y_col]), xytext=(5, 0), textcoords='offset points', fontsize=7, alpha=0.7)
            else:
                xerr = error_arrays(anytime, x_col, x_low_col, x_high_col)
                yerr = error_arrays(anytime, y_col, y_low_col, y_high_col)
                ax.errorbar(anytime[x_col], anytime[y_col], xerr=xerr, yerr=yerr, marker="^", linestyle="-", label="Anytime SC")

        title_suffix = f" ({dataset})" if dataset != "all" else ""
        ax.set_title(f"Accuracy vs Compute{title_suffix}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
