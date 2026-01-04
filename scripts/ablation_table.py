#!/usr/bin/env python
import argparse
import os
import pandas as pd

def load_summary(path, run_group=None, latest_group=False):
    df = pd.read_csv(path)
    if df.empty:
        return df
    if latest_group and "run_group" in df.columns:
        latest = df["run_group"].astype(str).max()
        df = df[df["run_group"] == latest]
    if run_group and "run_group" in df.columns:
        df = df[df["run_group"] == run_group]
    return df

def first_row(df):
    if df.empty:
        return None
    return df.iloc[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/summaries/summary_grouped.csv")
    parser.add_argument("--output", type=str, default="outputs/summaries/ablation_table.csv")
    parser.add_argument("--run_group", type=str)
    parser.add_argument("--latest_group", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Missing input file: {args.input}")

    df = load_summary(args.input, run_group=args.run_group, latest_group=args.latest_group)
    if df.empty:
        raise SystemExit("No rows available for ablation table.")

    records = []

    # Allocation ablation for anytime_sc
    anytime = df[df["method"] == "anytime_sc"]
    if not anytime.empty and "allocation" in anytime.columns:
        for (dataset, budget, delta), g in anytime.groupby(["dataset", "budget", "delta"], dropna=False):
            ucb = g[g["allocation"] == "ucb"]
            uniform = g[g["allocation"] == "uniform"]
            if ucb.empty or uniform.empty:
                continue
            a = first_row(ucb)
            b = first_row(uniform)
            records.append({
                "ablation": "allocation_ucb_vs_uniform",
                "dataset": dataset,
                "method_a": "anytime_sc",
                "method_b": "anytime_sc",
                "allocation_a": "ucb",
                "allocation_b": "uniform",
                "budget": budget,
                "delta": delta,
                "mean_accuracy_a": a["mean_accuracy"],
                "mean_accuracy_b": b["mean_accuracy"],
                "delta_accuracy": a["mean_accuracy"] - b["mean_accuracy"],
                "mean_tokens_a": a["mean_avg_tokens"],
                "mean_tokens_b": b["mean_avg_tokens"],
                "delta_tokens": a["mean_avg_tokens"] - b["mean_avg_tokens"],
            })

    # Allocation ablation for global_anytime_sc (vs uniform)
    global_df = df[df["method"] == "global_anytime_sc"]
    if not global_df.empty and "allocation" in global_df.columns:
        budget_col = "global_budget_tokens" if "global_budget_tokens" in global_df.columns else "budget"
        group_cols = ["dataset", budget_col, "policy", "init_k", "max_samples_per_item"]
        for keys, g in global_df.groupby(group_cols, dropna=False):
            dataset, budget, policy, init_k, max_k = keys
            uniform = g[g["allocation"] == "uniform"]
            if uniform.empty:
                continue
            base = first_row(uniform)
            for _, row in g.iterrows():
                if row.get("allocation") == "uniform":
                    continue
                records.append({
                    "ablation": "global_allocation_vs_uniform",
                    "dataset": dataset,
                    "method_a": "global_anytime_sc",
                    "method_b": "global_anytime_sc",
                    "allocation_a": row.get("allocation"),
                    "allocation_b": "uniform",
                    "budget": budget,
                    "policy": policy,
                    "init_k": init_k,
                    "max_samples_per_item": max_k,
                    "mean_accuracy_a": row["mean_accuracy"],
                    "mean_accuracy_b": base["mean_accuracy"],
                    "delta_accuracy": row["mean_accuracy"] - base["mean_accuracy"],
                    "mean_tokens_a": row["mean_total_tokens_sum"] if "mean_total_tokens_sum" in row else row["mean_avg_tokens"],
                    "mean_tokens_b": base["mean_total_tokens_sum"] if "mean_total_tokens_sum" in base else base["mean_avg_tokens"],
                    "delta_tokens": (
                        row["mean_total_tokens_sum"] - base["mean_total_tokens_sum"]
                        if "mean_total_tokens_sum" in row and "mean_total_tokens_sum" in base
                        else row["mean_avg_tokens"] - base["mean_avg_tokens"]
                    ),
                })

    # Matched-budget fixed-N vs anytime (per budget)
    fixed_sc = df[df["method"] == "self_consistency"]
    fixed_bon = df[df["method"] == "best_of_n"]
    if not anytime.empty:
        for _, a in anytime.iterrows():
            dataset = a.get("dataset")
            budget = a.get("budget")
            delta = a.get("delta")
            allocation = a.get("allocation")
            policy = a.get("policy")

            if pd.isna(budget):
                continue

            sc_matches = fixed_sc[(fixed_sc["dataset"] == dataset) & (fixed_sc["budget"] == budget)]
            if policy in sc_matches.get("policy", pd.Series()).values:
                sc_matches = sc_matches[sc_matches["policy"] == policy]
            for _, b in sc_matches.iterrows():
                records.append({
                    "ablation": "fixed_sc_vs_anytime",
                    "dataset": dataset,
                    "method_a": "anytime_sc",
                    "method_b": "self_consistency",
                    "allocation_a": allocation,
                    "budget": budget,
                    "delta": delta,
                    "policy": policy,
                    "n_b": b.get("n"),
                    "mean_accuracy_a": a["mean_accuracy"],
                    "mean_accuracy_b": b["mean_accuracy"],
                    "delta_accuracy": a["mean_accuracy"] - b["mean_accuracy"],
                    "mean_tokens_a": a["mean_avg_tokens"],
                    "mean_tokens_b": b["mean_avg_tokens"],
                    "delta_tokens": a["mean_avg_tokens"] - b["mean_avg_tokens"],
                })

            bon_matches = fixed_bon[(fixed_bon["dataset"] == dataset) & (fixed_bon["budget"] == budget)]
            if policy in bon_matches.get("policy", pd.Series()).values:
                bon_matches = bon_matches[bon_matches["policy"] == policy]
            for _, b in bon_matches.iterrows():
                records.append({
                    "ablation": "fixed_bon_vs_anytime",
                    "dataset": dataset,
                    "method_a": "anytime_sc",
                    "method_b": "best_of_n",
                    "allocation_a": allocation,
                    "budget": budget,
                    "delta": delta,
                    "policy": policy,
                    "n_b": b.get("n"),
                    "mean_accuracy_a": a["mean_accuracy"],
                    "mean_accuracy_b": b["mean_accuracy"],
                    "delta_accuracy": a["mean_accuracy"] - b["mean_accuracy"],
                    "mean_tokens_a": a["mean_avg_tokens"],
                    "mean_tokens_b": b["mean_avg_tokens"],
                    "delta_tokens": a["mean_avg_tokens"] - b["mean_avg_tokens"],
                })

    # Early-stop vs fixed-N (same n, policy)
    sc_es = df[df["method"] == "self_consistency_early_stop"]
    bon_es = df[df["method"] == "best_of_n_early_stop"]

    for _, a in sc_es.iterrows():
        matches = fixed_sc[(fixed_sc["dataset"] == a.get("dataset")) & (fixed_sc["n"] == a.get("n"))]
        if a.get("policy") in matches.get("policy", pd.Series()).values:
            matches = matches[matches["policy"] == a.get("policy")]
        for _, b in matches.iterrows():
            records.append({
                "ablation": "sc_early_stop_vs_fixed",
                "dataset": a.get("dataset"),
                "method_a": "self_consistency_early_stop",
                "method_b": "self_consistency",
                "n_a": a.get("n"),
                "n_b": b.get("n"),
                "policy": a.get("policy"),
                "mean_accuracy_a": a["mean_accuracy"],
                "mean_accuracy_b": b["mean_accuracy"],
                "delta_accuracy": a["mean_accuracy"] - b["mean_accuracy"],
                "mean_tokens_a": a["mean_avg_tokens"],
                "mean_tokens_b": b["mean_avg_tokens"],
                "delta_tokens": a["mean_avg_tokens"] - b["mean_avg_tokens"],
            })

    for _, a in bon_es.iterrows():
        matches = fixed_bon[(fixed_bon["dataset"] == a.get("dataset")) & (fixed_bon["n"] == a.get("n"))]
        if a.get("policy") in matches.get("policy", pd.Series()).values:
            matches = matches[matches["policy"] == a.get("policy")]
        for _, b in matches.iterrows():
            records.append({
                "ablation": "bon_early_stop_vs_fixed",
                "dataset": a.get("dataset"),
                "method_a": "best_of_n_early_stop",
                "method_b": "best_of_n",
                "n_a": a.get("n"),
                "n_b": b.get("n"),
                "policy": a.get("policy"),
                "mean_accuracy_a": a["mean_accuracy"],
                "mean_accuracy_b": b["mean_accuracy"],
                "delta_accuracy": a["mean_accuracy"] - b["mean_accuracy"],
                "mean_tokens_a": a["mean_avg_tokens"],
                "mean_tokens_b": b["mean_avg_tokens"],
                "delta_tokens": a["mean_avg_tokens"] - b["mean_avg_tokens"],
            })

    if not records:
        raise SystemExit("No ablation comparisons could be formed from the grouped summary.")

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved ablation table to {args.output}")

if __name__ == "__main__":
    main()
