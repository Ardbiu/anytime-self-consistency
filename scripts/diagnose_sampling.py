import argparse
import glob
import json
import os
import sys
import pandas as pd
import numpy as np

def get_latest_run_id(files):
    if not files: return None
    import re
    run_ids = set()
    for f in files:
        base = os.path.basename(f)
        match = re.search(r"(\d{8}-\d{6}(?:_[0-9a-f]{6})?)", base)
        if match: run_ids.add(match.group(1))
    if not run_ids: return None
    return sorted(list(run_ids))[-1]

def diagnose(files):
    all_methods = []
    
    print(f"Diagnosing {len(files)} files...")
    print("-" * 60)
    
    for f in files:
        df = pd.read_json(f, lines=True)
        if df.empty:
            continue
            
        first = df.iloc[0]
        method = first.get("method")
        
        # Skip greedy
        if method == "greedy":
            continue
            
        print(f"File: {os.path.basename(f)}")
        print(f"Method: {method} (n={first.get('n')}, b={first.get('budget_tokens')})")
        
        # Extract unique frac with a consistent definition:
        # unique_candidate_frac = (#unique final answers) / (num_candidates)
        unique_fracs = []
        all_same_examples = []

        for idx, row in df.iterrows():
            extra = row.get("extra", {})
            candidates = None

            if isinstance(extra, dict):
                candidates = extra.get("candidates")

            if not candidates and "steps" in row and isinstance(row["steps"], list):
                candidates = [s.get("raw_val", s.get("answer")) for s in row["steps"]]

            unique_count = None
            unique_frac = None

            if candidates is not None:
                num_candidates = len(candidates)
                if num_candidates == 0:
                    continue
                unique_count = len(set(candidates))
                unique_frac = unique_count / num_candidates
            else:
                if not isinstance(extra, dict):
                    continue
                unique_frac = extra.get("unique_candidate_frac")
                num_candidates = extra.get("num_candidates")
                if unique_frac is None or num_candidates in (None, 0):
                    continue
                unique_count = int(round(unique_frac * num_candidates))

            unique_fracs.append(unique_frac)
            if unique_count == 1:
                all_same_examples.append(row.get("example_id"))

        if unique_fracs:
            mean_div = np.mean(unique_fracs)
            print(f"  Mean Unique Candidate Frac: {mean_div:.4f}")
            print(f"  Min Unique Candidate Frac: {np.min(unique_fracs):.4f}")
            print(f"  All Candidates Identical (Count): {len(all_same_examples)} / {len(df)}")
            if all_same_examples:
                print(f"  Sample degenerate IDs: {all_same_examples[:5]}")
            
            if mean_div < 0.1:
                print("  [WARNING] Very low diversity! Sampling might be broken.")
            else:
                print("  [PASS] Diversity looks reasonable.")
        else:
            print("  [WARNING] No diversity metrics found in records.")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/runs")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.dir, "*.jsonl"))
    
    if args.latest:
        lid = get_latest_run_id(files)
        if lid:
            files = [f for f in files if lid in f]
            print(f"Filtering to latest run: {lid}")
        else:
            print("No run ID found.")
            return

    diagnose(files)

if __name__ == "__main__":
    main()
