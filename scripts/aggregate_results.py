import argparse
import pandas as pd
import json
import glob
import os
import sys

def get_latest_run_id(files):
    """
    Extracts timestamps/run_ids from filenames and returns the latest one.
    Assumes filenames end with ..._{run_id}.jsonl
    """
    if not files:
        return None
    
    # Try to extract the last segment before extension
    # Filename format: {dataset}_{method}_{params}_{run_id}.jsonl
    # Run ID format: YYYYMMDD-HHMMSS_hash (new) or YYYYMMDD-HHMMSS (old)
    import re
    run_ids = set()
    for f in files:
        base = os.path.basename(f)
        # Search for timestamp pattern
        match = re.search(r"(\d{8}-\d{6}(?:_[0-9a-f]{6})?)", base)
        if match:
            run_ids.add(match.group(1))
        else:
            # Fallback: simple split (likely fails for complex IDs)
            pass
        
    if not run_ids:
        return None
        
    # Sort them. Timestamps sort lexicographically correctly.
    return sorted(list(run_ids))[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory containing JSONL files")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--run_id", type=str, help="Specific run_id to aggregate")
    parser.add_argument("--latest", action="store_true", help="Aggregate only the latest run_id found in input dir")
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

    if args.latest:
        target_run_id = get_latest_run_id(all_files)
        print(f"Detected latest run_id: {target_run_id}")
    
    if target_run_id:
        target_files = [f for f in all_files if target_run_id in f]
        if not target_files:
            print(f"Error: No files found matching run_id: {target_run_id}")
            sys.exit(1)
        print(f"Aggregating {len(target_files)} files for run_id: {target_run_id}")
    else:
        print(f"Aggregating ALL {len(all_files)} files in directory. Use --latest or --run_id to filter.")

    records = []
    
    import numpy as np
    
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

    for fpath in target_files:
        try:
            # We assume one config per file, but read all lines to be safe
            df = pd.read_json(fpath, lines=True)
        except ValueError:
            print(f"Skipping empty or invalid file: {fpath}")
            continue
            
        if len(df) == 0:
            continue
            
        # Group by method configuration (should be unique per file usually, but robust grouping is better)
        # However, to do CIs, we need the raw arrays.
        # Let's verify if the file contains mixed methods (unlikely with current runner).
        # We'll treat the file as one "experiment group".
        
        first = df.iloc[0]
        
        # Extract diversity metrics from 'extra'
        # 'extra' is a dict column.
        unique_fracs = []
        if "extra" in df.columns:
            for x in df["extra"]:
                if isinstance(x, dict):
                    uf = x.get("unique_candidate_frac")
                    if uf is not None:
                        unique_fracs.append(uf)
        
        mean_unique_frac = np.mean(unique_fracs) if unique_fracs else None
        
        acc_low, acc_high = bootstrap_ci(df["is_correct"].dropna().astype(float))
        tokens_low, tokens_high = bootstrap_ci(df["total_tokens"].dropna())
        
        row = {
            "dataset": first.get("dataset"),
            "model_name": first.get("model_name"),
            "method": first.get("method"),
            "n": first.get("n"),
            "budget": first.get("budget_tokens"),
            "delta": first.get("delta"),
            "allocation": first.get("allocation"),
            "policy": first.get("policy"), 
            "run_id": first.get("run_id", "unknown"),
            
            "accuracy": df["is_correct"].mean(),
            "accuracy_ci_low": acc_low,
            "accuracy_ci_high": acc_high,
            
            "avg_tokens": df["total_tokens"].mean(),
            "avg_tokens_ci_low": tokens_low,
            "avg_tokens_ci_high": tokens_high,
            
            "avg_time_s": df["time_s"].mean(),
            "unique_candidate_frac": mean_unique_frac,
            "count": len(df)
        }
        
        records.append(row)
        
    if not records:
        print("No valid records found.")
        return

    summary_df = pd.DataFrame(records)
    
    # Sort for niceness
    cols = [
        "dataset", "method", "model_name", "n", "budget", "delta", "allocation", 
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "avg_tokens", "avg_tokens_ci_low", "avg_tokens_ci_high",
        "unique_candidate_frac", "avg_time_s", "count", "run_id"
    ]
    # Filter to existing cols
    cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df[cols]
    
    print(summary_df)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print(f"Saved summary to {args.output}")

if __name__ == "__main__":
    main()
