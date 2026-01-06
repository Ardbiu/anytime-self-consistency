
import json
import glob
import pandas as pd
import os
import numpy as np

def bootstrap_ci(accuracies, n_boot=1000, ci=95):
    if len(accuracies) < 2:
        return np.mean(accuracies), np.mean(accuracies)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(accuracies, size=len(accuracies), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

files = glob.glob('outputs/runs/*20260105*.jsonl')
print(f"Found {len(files)} files")

results = []
for f in files:
    try:
        with open(f) as fp:
            lines = [json.loads(line) for line in fp if line.strip()]
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

    if not lines:
        continue
        
    first = lines[0]
    method = first.get('method', 'unknown')
    n = first.get('n', '')
    budget = first.get('budget_tokens', '') or first.get('global_budget_tokens', '')
    if method == 'anytime_sc':
        budget = first.get('budget', '')
    
    # Calculate accuracy
    is_correct = [1 if x.get('is_correct') else 0 for x in lines]
    accuracy = np.mean(is_correct) * 100
    
    # Calculate tokens
    total_tokens = sum(x.get('total_tokens', 0) for x in lines)
    avg_tokens = total_tokens / len(lines)
    
    results.append({
        'method': method,
        'config': f"{method} (n={n}, b={budget})",
        'mean_accuracy': accuracy,
        'mean_avg_tokens': avg_tokens,
        'samples': len(lines),
        # Helper cols
        'n': n,
        'budget': budget
    })

raw_df = pd.DataFrame(results)

# Aggregate by config
df = raw_df.groupby(['method', 'config', 'n', 'budget']).agg({
    'mean_accuracy': 'mean',
    'mean_avg_tokens': 'mean',
    'samples': 'sum'
}).reset_index()

print("\n=== RESULTS SUMMARY ===")
print(df.sort_values(['method', 'mean_avg_tokens']).to_string())

# Add missing columns for plotter - explicit CIs to avoid shape errors
df['dataset'] = 'gsm8k'
df['run_group'] = 'manual_analysis'

# Zero-width CIs
df['accuracy_ci_low'] = df['mean_accuracy']
df['accuracy_ci_high'] = df['mean_accuracy']
df['tokens_ci_low'] = df['mean_avg_tokens']
df['tokens_ci_high'] = df['mean_avg_tokens']

output_path = 'outputs/summaries/summary_grouped.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved compatible summary to {output_path}")

# Simple Pareto check
print("\n=== PARETO CHECK ===")
greedy = df[df['method'] == 'greedy']
greedy_acc = greedy['mean_accuracy'].values[0] if not greedy.empty else 0
greedy_tok = greedy['mean_avg_tokens'].values[0] if not greedy.empty else 0

sc = df[df['method'] == 'self_consistency'].sort_values('mean_avg_tokens')
global_sc = df[df['method'] == 'global_anytime_sc'].sort_values('mean_avg_tokens')

print(f"Greedy: {greedy_acc:.1f}% acc @ {greedy_tok:.0f} tokens")

print("\nSelf-Consistency:")
for _, row in sc.iterrows():
    print(f"  {row['mean_avg_tokens']:.0f} tok -> {row['mean_accuracy']:.1f}%")

print("\nGlobal Anytime SC:")
for _, row in global_sc.iterrows():
    print(f"  {row['mean_avg_tokens']:.0f} tok -> {row['mean_accuracy']:.1f}%")
