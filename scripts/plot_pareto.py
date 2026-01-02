import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input summary CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output plot PNG file")
    parser.add_argument("--run_id", type=str, help="Specific run_id to plot")
    parser.add_argument("--latest", action="store_true", help="Plot only the latest run_id found in summary")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
        
    df = pd.read_csv(args.input)
    if df.empty:
        print("Error: Empty summary CSV.")
        sys.exit(1)
        
    # Filtering
    if args.latest:
        if "run_id" not in df.columns:
            print("Warning: 'run_id' column not found in summary. Cannot filter by latest.")
        else:
            # find latest string.
            # Assuming timestamps format YYYYMMDD... works lexicographically
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
        
    # Check if multiple runs still exist (if user didn't filter)
    if "run_id" in df.columns:
        unique_runs = df["run_id"].unique()
        if len(unique_runs) > 1:
            print(f"Warning: Plotting data from {len(unique_runs)} different run_ids. Use --latest or --run_id to filter.")
    
    plt.figure(figsize=(10, 6))
    
    # 1. Greedy
    greedy = df[df["method"] == "greedy"]
    if not greedy.empty:
        plt.scatter(greedy["avg_tokens"], greedy["accuracy"], label="Greedy", marker="x", s=100, color="black", zorder=5)
        
    # 2. SC
    sc = df[df["method"] == "self_consistency"].sort_values("avg_tokens")
    if not sc.empty:
        # Check if we have multiple N values per n (mixed runs)?
        # If filtered correctly, we should be fine.
        plt.plot(sc["avg_tokens"], sc["accuracy"], marker="o", linestyle="-", label="Self-Consistency")
        for _, row in sc.iterrows():
            plt.annotate(f"n={int(row['n'])}", (row['avg_tokens'], row['accuracy']), xytext=(0, 5), textcoords='offset points', fontsize=8)
        
    # 3. Best of N
    bon = df[df["method"] == "best_of_n"].sort_values("avg_tokens")
    if not bon.empty:
        plt.plot(bon["avg_tokens"], bon["accuracy"], marker="s", linestyle="--", label="Best-of-N")
        for _, row in bon.iterrows():
            plt.annotate(f"n={int(row['n'])}", (row['avg_tokens'], row['accuracy']), xytext=(0, -10), textcoords='offset points', fontsize=8)
        
    # 4. Anytime
    anytime = df[df["method"] == "anytime_sc"]
    if not anytime.empty:
        # Group by allocation if present
        if "allocation" in anytime.columns:
            allocs = anytime["allocation"].unique()
            for alloc in allocs:
                if pd.isna(alloc): continue
                sub = anytime[anytime["allocation"] == alloc].sort_values("avg_tokens")
                label_txt = f"Anytime ({alloc})"
                plt.plot(sub["avg_tokens"], sub["accuracy"], marker="^", label=label_txt)
                for _, row in sub.iterrows():
                    # Annotate budget/delta
                    lbl = f"b={int(row['budget'])}"
                    if "delta" in row and not pd.isna(row['delta']):
                        lbl += f", d={row['delta']}"
                    plt.annotate(lbl, (row['avg_tokens'], row['accuracy']), xytext=(5, 0), textcoords='offset points', fontsize=7, alpha=0.7)
        else:
             sub = anytime.sort_values("avg_tokens")
             plt.plot(sub["avg_tokens"], sub["accuracy"], marker="^", label="Anytime SC")

    plt.xlabel("Average Total Tokens per Example")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Compute (Pareto Frontier)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
