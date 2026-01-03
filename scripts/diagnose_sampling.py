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
        
        # Extract unique frac
        unique_fracs = []
        zero_diversity_examples = []
        
        for idx, row in df.iterrows():
            extra = row.get("extra", {})
            if not isinstance(extra, dict): continue
            
            uf = extra.get("unique_candidate_frac")
            if uf is not None:
                unique_fracs.append(uf)
                if uf <= (1.0 / row.get("n", 100)) + 0.01: 
                    # If roughly 1/N, it usually means 1 unique candidate out of N => diversity 0.
                    # Actually frac is unique/total. If 1 unique, frac is 1/N. 
                    # But if N is large, 1/N is small.
                    # Simpler check: num_candidates
                    pass
                
                # Check for absolute zero diversity (only 1 unique answer)
                # If unique_frac * num_candidates approx 1
                try:
                    candidates = extra.get("candidates", [])
                    if not candidates and "steps" in row:
                        # Anytime steps
                        candidates = [s.get("answer") for s in row["steps"]]
                    
                    if candidates:
                        unique_set = set(candidates)
                        if len(unique_set) == 1 and len(candidates) > 1:
                           zero_diversity_examples.append(row.get("example_id"))
                except:
                    pass

        if unique_fracs:
            mean_div = np.mean(unique_fracs)
            print(f"  Mean Unique Candidate Frac: {mean_div:.4f}")
            print(f"  Min Diversity: {np.min(unique_fracs):.4f}")
            print(f"  Zero Diversity Examples (All Same): {len(zero_diversity_examples)} / {len(df)}")
            if zero_diversity_examples:
                print(f"  Sample degenerate IDs: {zero_diversity_examples[:5]}")
            
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
