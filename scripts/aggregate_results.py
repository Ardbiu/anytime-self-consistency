import argparse
import pandas as pd
import json
import glob
import os
import sys
import numpy as np

def get_latest_run_id(files):
    if not files:
        return None
    import re
    run_ids = set()
    for f in files:
        base = os.path.basename(f)
        match = re.search(r"(\d{8}-\d{6}(?:_[0-9a-f]{6})?)", base)
        if match:
            run_ids.add(match.group(1))
    if not run_ids:
        return None
    return sorted(list(run_ids))[-1]

def read_first_record(path):
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except Exception:
        return {}
    return {}

def get_latest_run_group(files):
    run_groups = set()
    for f in files:
        rec = read_first_record(f)
        rg = rec.get("run_group")
        if rg:
            run_groups.add(rg)
    if not run_groups:
        return None
    return sorted(run_groups)[-1]

def bootstrap_ci(data, n_boot=1000, ci=95):
    if len(data) < 2:
        return np.nan, np.nan
    boot_means = []
    data_arr = np.array(data)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(data_arr, size=len(data_arr), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/runs", help="Input directory containing JSONL files")
    parser.add_argument("--output", type=str, help="Legacy per-run output CSV path")
    parser.add_argument("--output_per_run", type=str, default="outputs/summaries/summary_per_run.csv")
    parser.add_argument("--output_grouped", type=str, default="outputs/summaries/summary_grouped.csv")
    parser.add_argument("--run_id", type=str, help="Specific run_id to aggregate")
    parser.add_argument("--latest", action="store_true", help="Aggregate only the latest run_id found in input dir")
    parser.add_argument("--run_group", type=str, help="Specific run_group to aggregate")
    parser.add_argument("--latest_group", action="store_true", help="Aggregate only the latest run_group found in input dir")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for CI estimation")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist.")
        sys.exit(1)

    all_files = glob.glob(os.path.join(args.input, "*.jsonl"))
    if not all_files:
        print(f"Warning: No JSONL files found in {args.input}")
        return

    target_files = all_files
    target_run_id = args.run_id
    target_run_group = args.run_group

    if args.latest_group:
        target_run_group = get_latest_run_group(all_files)
        print(f"Detected latest run_group: {target_run_group}")

    if args.latest:
        target_run_id = get_latest_run_id(all_files)
        print(f"Detected latest run_id: {target_run_id}")

    if target_run_group:
        filtered = []
        for f in all_files:
            rec = read_first_record(f)
            if rec.get("run_group") == target_run_group:
                filtered.append(f)
        target_files = filtered
        if not target_files:
            print(f"Error: No files found matching run_group: {target_run_group}")
            sys.exit(1)
        print(f"Aggregating {len(target_files)} files for run_group: {target_run_group}")
    elif target_run_id:
        target_files = [f for f in all_files if target_run_id in f]
        if not target_files:
            print(f"Error: No files found matching run_id: {target_run_id}")
            sys.exit(1)
        print(f"Aggregating {len(target_files)} files for run_id: {target_run_id}")
    else:
        print(f"Aggregating ALL {len(all_files)} files in directory.")

    per_run_records = []
    per_example_rows = []

    for fpath in target_files:
        try:
            df = pd.read_json(fpath, lines=True)
        except ValueError:
            print(f"Skipping empty or invalid file: {fpath}")
            continue

        if len(df) == 0:
            continue

        first = df.iloc[0]
        run_id = first.get("run_id", "unknown")
        run_group = first.get("run_group") or run_id
        seed = first.get("seed")

        unique_fracs = []
        if "extra" in df.columns:
            for x in df["extra"]:
                if isinstance(x, dict):
                    uf = x.get("unique_candidate_frac")
                    if uf is not None:
                        unique_fracs.append(uf)

        mean_unique_frac = float(np.mean(unique_fracs)) if unique_fracs else np.nan

        acc_vals = df["is_correct"].dropna().astype(float).tolist()
        tok_vals = df["total_tokens"].dropna().tolist()
        total_tokens_sum = df["total_tokens"].dropna().sum()
        total_time_sum = df["time_s"].dropna().sum()
        correct_count = float(df["is_correct"].sum())
        tokens_per_correct = total_tokens_sum / correct_count if correct_count > 0 else np.nan
        time_per_correct = total_time_sum / correct_count if correct_count > 0 else np.nan
        accuracy_per_second = correct_count / total_time_sum if total_time_sum > 0 else np.nan
        acc_low, acc_high = bootstrap_ci(acc_vals, n_boot=args.bootstrap)
        tokens_low, tokens_high = bootstrap_ci(tok_vals, n_boot=args.bootstrap)

        per_run_records.append({
            "dataset": first.get("dataset"),
            "model_name": first.get("model_name"),
            "method": first.get("method"),
            "n": first.get("n"),
            "budget": first.get("budget_tokens"),
            "global_budget_tokens": first.get("global_budget_tokens", first.get("budget_tokens")),
            "delta": first.get("delta"),
            "allocation": first.get("allocation"),
            "policy": first.get("policy"),
            "verifier_model_name": first.get("verifier_model_name"),
            "verifier_max_new_tokens": first.get("verifier_max_new_tokens"),
            "verifier_task": first.get("verifier_task"),
            "batched": first.get("batched"),
            "batch_size": first.get("batch_size"),
            "allow_unseeded_batch": first.get("allow_unseeded_batch"),
            "init_k": first.get("init_k"),
            "max_samples_per_item": first.get("max_samples_per_item"),
            "per_example_budget_tokens": first.get("per_example_budget_tokens"),
            "ucb_c": first.get("ucb_c"),
            "run_id": run_id,
            "run_group": run_group,
            "seed": seed,
            "accuracy": df["is_correct"].mean(),
            "accuracy_ci_low": acc_low,
            "accuracy_ci_high": acc_high,
            "avg_tokens": df["total_tokens"].mean(),
            "avg_tokens_ci_low": tokens_low,
            "avg_tokens_ci_high": tokens_high,
            "total_tokens_sum": total_tokens_sum,
            "total_time_sum": total_time_sum,
            "correct_count": correct_count,
            "tokens_per_correct": tokens_per_correct,
            "time_per_correct": time_per_correct,
            "accuracy_per_second": accuracy_per_second,
            "avg_time_s": df["time_s"].mean(),
            "unique_candidate_frac": mean_unique_frac,
            "count": len(df),
        })

        for _, row in df.iterrows():
            extra = row.get("extra", {})
            uf = None
            if isinstance(extra, dict):
                uf = extra.get("unique_candidate_frac")
            per_example_rows.append({
                "dataset": row.get("dataset"),
                "model_name": row.get("model_name"),
                "method": row.get("method"),
                "n": row.get("n"),
                "budget": row.get("budget_tokens"),
                "global_budget_tokens": row.get("global_budget_tokens", row.get("budget_tokens")),
                "delta": row.get("delta"),
                "allocation": row.get("allocation"),
                "policy": row.get("policy"),
                "verifier_model_name": row.get("verifier_model_name"),
                "verifier_max_new_tokens": row.get("verifier_max_new_tokens"),
                "verifier_task": row.get("verifier_task"),
                "batched": row.get("batched"),
                "batch_size": row.get("batch_size"),
                "allow_unseeded_batch": row.get("allow_unseeded_batch"),
                "init_k": row.get("init_k"),
                "max_samples_per_item": row.get("max_samples_per_item"),
                "per_example_budget_tokens": row.get("per_example_budget_tokens"),
                "ucb_c": row.get("ucb_c"),
                "run_id": row.get("run_id"),
                "run_group": row.get("run_group") or run_id,
                "seed": row.get("seed", seed),
                "is_correct": row.get("is_correct"),
                "total_tokens": row.get("total_tokens"),
                "time_s": row.get("time_s"),
                "unique_candidate_frac": uf,
            })

    if not per_run_records:
        print("No valid records found.")
        return

    per_run_df = pd.DataFrame(per_run_records)
    per_run_cols = [
        "dataset", "method", "model_name", "n", "budget", "global_budget_tokens", "delta", "allocation", "policy",
        "verifier_model_name", "verifier_max_new_tokens", "verifier_task",
        "batched", "batch_size", "allow_unseeded_batch",
        "init_k", "max_samples_per_item", "per_example_budget_tokens", "ucb_c",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "avg_tokens", "avg_tokens_ci_low", "avg_tokens_ci_high",
        "total_tokens_sum", "total_time_sum", "correct_count", "tokens_per_correct",
        "time_per_correct", "accuracy_per_second", "unique_candidate_frac",
        "avg_time_s", "count", "run_id", "run_group", "seed"
    ]
    per_run_cols = [c for c in per_run_cols if c in per_run_df.columns]
    per_run_df = per_run_df[per_run_cols]

    output_per_run = args.output or args.output_per_run
    os.makedirs(os.path.dirname(output_per_run), exist_ok=True)
    per_run_df.to_csv(output_per_run, index=False)
    print(f"Saved per-run summary to {output_per_run}")

    grouped_df = pd.DataFrame(per_example_rows)
    if grouped_df.empty:
        print("No per-example rows available for grouped summary.")
        return

    grouped_df["seed"] = grouped_df["seed"].fillna(-1)

    group_cols = [
        "run_group", "dataset", "model_name", "method", "n", "budget", "global_budget_tokens", "delta", "allocation", "policy",
        "verifier_model_name", "verifier_max_new_tokens", "verifier_task",
        "batched", "batch_size", "allow_unseeded_batch",
        "init_k", "max_samples_per_item", "per_example_budget_tokens", "ucb_c"
    ]
    grouped_records = []
    for keys, gdf in grouped_df.groupby(group_cols, dropna=False):
        seed_stats = gdf.groupby("seed").agg(
            accuracy_mean=("is_correct", "mean"),
            tokens_mean=("total_tokens", "mean"),
            time_mean=("time_s", "mean"),
            unique_mean=("unique_candidate_frac", "mean"),
        )
        seed_accs = seed_stats["accuracy_mean"].dropna().tolist()
        seed_tokens = seed_stats["tokens_mean"].dropna().tolist()
        seed_times = seed_stats["time_mean"].dropna().tolist()
        seed_unique = seed_stats["unique_mean"].dropna().tolist()
        seed_total_tokens = gdf.groupby("seed")["total_tokens"].sum()
        seed_total_times = gdf.groupby("seed")["time_s"].sum()
        seed_correct = gdf.groupby("seed")["is_correct"].sum()
        seed_tokens_per_correct = (seed_total_tokens / seed_correct.replace(0, np.nan)).dropna().tolist()
        seed_time_per_correct = (seed_total_times / seed_correct.replace(0, np.nan)).dropna().tolist()
        seed_accuracy_per_second = (seed_correct / seed_total_times.replace(0, np.nan)).dropna().tolist()
        seed_total_tokens_list = seed_total_tokens.dropna().tolist()

        acc_low, acc_high = bootstrap_ci(gdf["is_correct"].dropna().astype(float).tolist(), n_boot=args.bootstrap)
        tok_low, tok_high = bootstrap_ci(gdf["total_tokens"].dropna().tolist(), n_boot=args.bootstrap)
        time_low, time_high = bootstrap_ci(gdf["time_s"].dropna().tolist(), n_boot=args.bootstrap)
        uniq_low, uniq_high = bootstrap_ci(gdf["unique_candidate_frac"].dropna().tolist(), n_boot=args.bootstrap)
        sum_low, sum_high = bootstrap_ci(seed_total_tokens_list, n_boot=args.bootstrap)

        record = dict(zip(group_cols, keys))
        record.update({
            "seed_count": len(seed_stats),
            "mean_accuracy": float(np.mean(seed_accs)) if seed_accs else np.nan,
            "std_accuracy": float(np.std(seed_accs, ddof=1)) if len(seed_accs) > 1 else 0.0,
            "accuracy_ci_low": acc_low,
            "accuracy_ci_high": acc_high,
            "mean_avg_tokens": float(np.mean(seed_tokens)) if seed_tokens else np.nan,
            "std_avg_tokens": float(np.std(seed_tokens, ddof=1)) if len(seed_tokens) > 1 else 0.0,
            "tokens_ci_low": tok_low,
            "tokens_ci_high": tok_high,
            "mean_avg_time_s": float(np.mean(seed_times)) if seed_times else np.nan,
            "std_avg_time_s": float(np.std(seed_times, ddof=1)) if len(seed_times) > 1 else 0.0,
            "time_ci_low": time_low,
            "time_ci_high": time_high,
            "mean_total_tokens_sum": float(np.mean(seed_total_tokens_list)) if seed_total_tokens_list else np.nan,
            "std_total_tokens_sum": float(np.std(seed_total_tokens_list, ddof=1)) if len(seed_total_tokens_list) > 1 else 0.0,
            "total_tokens_sum_ci_low": sum_low,
            "total_tokens_sum_ci_high": sum_high,
            "mean_unique_candidate_frac": float(np.mean(seed_unique)) if seed_unique else np.nan,
            "unique_candidate_frac_ci_low": uniq_low,
            "unique_candidate_frac_ci_high": uniq_high,
            "mean_tokens_per_correct": float(np.mean(seed_tokens_per_correct)) if seed_tokens_per_correct else np.nan,
            "std_tokens_per_correct": float(np.std(seed_tokens_per_correct, ddof=1)) if len(seed_tokens_per_correct) > 1 else 0.0,
            "mean_time_per_correct": float(np.mean(seed_time_per_correct)) if seed_time_per_correct else np.nan,
            "std_time_per_correct": float(np.std(seed_time_per_correct, ddof=1)) if len(seed_time_per_correct) > 1 else 0.0,
            "mean_accuracy_per_second": float(np.mean(seed_accuracy_per_second)) if seed_accuracy_per_second else np.nan,
            "std_accuracy_per_second": float(np.std(seed_accuracy_per_second, ddof=1)) if len(seed_accuracy_per_second) > 1 else 0.0,
            "count": len(gdf),
        })
        grouped_records.append(record)

    grouped_summary_df = pd.DataFrame(grouped_records)
    grouped_cols = [
        "run_group", "dataset", "method", "model_name", "n", "budget", "global_budget_tokens", "delta", "allocation", "policy",
        "verifier_model_name", "verifier_max_new_tokens", "verifier_task",
        "batched", "batch_size", "allow_unseeded_batch",
        "init_k", "max_samples_per_item", "per_example_budget_tokens", "ucb_c",
        "mean_accuracy", "std_accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "mean_avg_tokens", "std_avg_tokens", "tokens_ci_low", "tokens_ci_high",
        "mean_avg_time_s", "std_avg_time_s", "time_ci_low", "time_ci_high",
        "mean_total_tokens_sum", "std_total_tokens_sum", "total_tokens_sum_ci_low", "total_tokens_sum_ci_high",
        "mean_unique_candidate_frac", "unique_candidate_frac_ci_low", "unique_candidate_frac_ci_high",
        "mean_tokens_per_correct", "std_tokens_per_correct",
        "mean_time_per_correct", "std_time_per_correct",
        "mean_accuracy_per_second", "std_accuracy_per_second",
        "seed_count", "count"
    ]
    grouped_cols = [c for c in grouped_cols if c in grouped_summary_df.columns]
    grouped_summary_df = grouped_summary_df[grouped_cols]

    os.makedirs(os.path.dirname(args.output_grouped), exist_ok=True)
    grouped_summary_df.to_csv(args.output_grouped, index=False)
    print(f"Saved grouped summary to {args.output_grouped}")

    global_df = grouped_summary_df[grouped_summary_df["method"] == "global_anytime_sc"].copy()
    if not global_df.empty:
        points_cols = [
            "run_group", "dataset", "method", "model_name", "budget", "global_budget_tokens",
            "allocation", "policy", "init_k", "max_samples_per_item", "per_example_budget_tokens", "ucb_c",
            "mean_accuracy", "accuracy_ci_low", "accuracy_ci_high",
            "mean_total_tokens_sum", "total_tokens_sum_ci_low", "total_tokens_sum_ci_high",
            "seed_count", "count",
        ]
        points_cols = [c for c in points_cols if c in global_df.columns]
        global_points = global_df[points_cols]
        points_path = "outputs/summaries/summary_global_points.csv"
        os.makedirs(os.path.dirname(points_path), exist_ok=True)
        global_points.to_csv(points_path, index=False)
        print(f"Saved global points to {points_path}")

        curve_records = []
        group_cols = ["run_group", "dataset", "model_name", "allocation", "policy", "init_k", "max_samples_per_item"]
        for keys, gdf in global_df.groupby(group_cols, dropna=False):
            gdf = gdf.dropna(subset=["mean_total_tokens_sum", "mean_accuracy"]).sort_values("mean_total_tokens_sum")
            if len(gdf) < 2:
                continue
            x = gdf["mean_total_tokens_sum"].to_numpy()
            y = gdf["mean_accuracy"].to_numpy()
            auc = float(np.trapezoid(y, x))
            record = dict(zip(group_cols, keys))
            record.update({
                "auc": auc,
                "max_tokens": float(np.max(x)),
                "n_points": int(len(gdf)),
            })
            curve_records.append(record)

        if curve_records:
            curve_df = pd.DataFrame(curve_records)
            curve_path = "outputs/summaries/summary_global_curve.csv"
            curve_df.to_csv(curve_path, index=False)
            print(f"Saved global curve summary to {curve_path}")

if __name__ == "__main__":
    main()
