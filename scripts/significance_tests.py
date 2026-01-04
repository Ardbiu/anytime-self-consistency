#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

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

def match_record(rec, dataset=None, method=None, n=None, budget=None, delta=None, allocation=None, policy=None):
    if dataset and rec.get("dataset") != dataset:
        return False
    if method and rec.get("method") != method:
        return False
    if n is not None and rec.get("n") != n:
        return False
    if budget is not None and rec.get("budget_tokens") != budget:
        return False
    if delta is not None and rec.get("delta") != delta:
        return False
    if allocation is not None and rec.get("allocation") != allocation:
        return False
    if policy is not None and rec.get("policy") != policy:
        return False
    return True

def load_matching_files(files, run_group, dataset, method, n, budget, delta, allocation, policy):
    selected = []
    for f in files:
        rec = read_first_record(f)
        if run_group and rec.get("run_group") != run_group:
            continue
        if not match_record(rec, dataset, method, n, budget, delta, allocation, policy):
            continue
        selected.append(f)
    return selected

def bootstrap_ci(diffs, n_boot=1000, ci=95):
    if len(diffs) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    diffs = np.array(diffs)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    p_low = np.mean(np.array(boot_means) <= 0)
    p_high = np.mean(np.array(boot_means) >= 0)
    p_val = min(1.0, 2 * min(p_low, p_high))
    return lower, upper, p_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/runs", help="Directory with JSONL files")
    parser.add_argument("--run_group", type=str, help="Run group to compare")
    parser.add_argument("--latest_group", action="store_true", help="Use latest run_group")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method_a", type=str, required=True)
    parser.add_argument("--method_b", type=str, required=True)
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "tokens"])
    parser.add_argument("--bootstrap", type=int, default=1000)

    parser.add_argument("--a_n", type=int)
    parser.add_argument("--a_budget", type=int)
    parser.add_argument("--a_delta", type=float)
    parser.add_argument("--a_allocation", type=str)
    parser.add_argument("--a_policy", type=str)

    parser.add_argument("--b_n", type=int)
    parser.add_argument("--b_budget", type=int)
    parser.add_argument("--b_delta", type=float)
    parser.add_argument("--b_allocation", type=str)
    parser.add_argument("--b_policy", type=str)

    parser.add_argument("--output", type=str, help="Optional CSV output path")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.dir, "*.jsonl"))
    if not files:
        print(f"Error: No JSONL files found in {args.dir}")
        sys.exit(1)

    run_group = args.run_group
    if args.latest_group:
        run_group = get_latest_run_group(files)
        if not run_group:
            print("Error: Could not determine latest run_group.")
            sys.exit(1)
        print(f"Using latest run_group: {run_group}")

    files_a = load_matching_files(
        files,
        run_group,
        args.dataset,
        args.method_a,
        args.a_n,
        args.a_budget,
        args.a_delta,
        args.a_allocation,
        args.a_policy,
    )
    files_b = load_matching_files(
        files,
        run_group,
        args.dataset,
        args.method_b,
        args.b_n,
        args.b_budget,
        args.b_delta,
        args.b_allocation,
        args.b_policy,
    )

    if not files_a:
        print("Error: No files found for method_a with the requested filters.")
        sys.exit(1)
    if not files_b:
        print("Error: No files found for method_b with the requested filters.")
        sys.exit(1)

    df_a = pd.concat([pd.read_json(f, lines=True) for f in files_a], ignore_index=True)
    df_b = pd.concat([pd.read_json(f, lines=True) for f in files_b], ignore_index=True)

    df_a = df_a[df_a["dataset"] == args.dataset]
    df_b = df_b[df_b["dataset"] == args.dataset]

    df_a = df_a.sort_values("run_id").drop_duplicates(subset=["dataset", "seed", "example_id"], keep="last")
    df_b = df_b.sort_values("run_id").drop_duplicates(subset=["dataset", "seed", "example_id"], keep="last")

    merged = df_a.merge(
        df_b,
        on=["dataset", "seed", "example_id"],
        suffixes=("_a", "_b"),
        how="inner"
    )

    if merged.empty:
        print("Error: No paired examples found between methods.")
        sys.exit(1)

    if args.metric == "accuracy":
        diffs = merged["is_correct_a"].astype(float) - merged["is_correct_b"].astype(float)
    else:
        diffs = merged["total_tokens_a"].astype(float) - merged["total_tokens_b"].astype(float)

    diffs = diffs.dropna().to_numpy()
    if len(diffs) < 2:
        print("Error: Not enough paired samples for bootstrap.")
        sys.exit(1)

    mean_diff = float(np.mean(diffs))
    ci_low, ci_high, p_val = bootstrap_ci(diffs, n_boot=args.bootstrap)

    result = {
        "dataset": args.dataset,
        "method_a": args.method_a,
        "method_b": args.method_b,
        "metric": args.metric,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_val,
        "n_pairs": int(len(diffs)),
        "run_group": run_group,
    }

    print(json.dumps(result, indent=2))

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([result]).to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
