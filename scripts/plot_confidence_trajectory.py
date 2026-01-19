#!/usr/bin/env python
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt

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

def entropy_from_counts(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent

def extract_entropy_trajectory(steps):
    counts = Counter()
    trajectory = []
    stop_t = None
    for step in steps:
        if step.get("raw_val") is not None:
            counts[str(step.get("raw_val"))] += 1
        trajectory.append(entropy_from_counts(counts))
        if stop_t is None and step.get("stop"):
            stop_t = step.get("t")
    return trajectory, stop_t

def pick_examples(records, pick, num_examples):
    if not records:
        return []
    with_steps = [r for r in records if isinstance(r.get("steps"), list) and r.get("steps")]
    if not with_steps:
        return []
    with_steps.sort(key=lambda r: len(r.get("steps", [])))

    if pick == "easy":
        return with_steps[:num_examples]
    if pick == "hard":
        return with_steps[-num_examples:]
    if pick == "both":
        easy = with_steps[:num_examples]
        hard = with_steps[-num_examples:]
        return easy + hard
    return with_steps[:num_examples]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/runs")
    parser.add_argument("--run_group", type=str)
    parser.add_argument("--latest_group", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--method", type=str, default="anytime_sc")
    parser.add_argument("--example_ids", type=str, help="Comma-separated example ids")
    parser.add_argument("--pick", type=str, default="both", choices=["easy", "hard", "both"])
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--output", type=str, default="outputs/plots/confidence_trajectory.png")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.dir, "*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found in {args.dir}")

    run_group = args.run_group
    if args.latest_group:
        run_group = get_latest_run_group(files)
        if not run_group:
            raise SystemExit("No run_group found to analyze.")

    selected = []
    for f in files:
        rec = read_first_record(f)
        if run_group and rec.get("run_group") != run_group:
            continue
        if args.dataset and rec.get("dataset") != args.dataset:
            continue
        if args.method and rec.get("method") != args.method:
            continue
        selected.append(f)

    if not selected:
        raise SystemExit("No files match the requested filters.")

    records = []
    for f in selected:
        with open(f, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if args.method and rec.get("method") != args.method:
                    continue
                if args.dataset and rec.get("dataset") != args.dataset:
                    continue
                records.append(rec)

    if not records:
        raise SystemExit("No records found after filtering.")

    if args.example_ids:
        ids = [x.strip() for x in args.example_ids.split(",") if x.strip()]
        selected_records = [r for r in records if r.get("example_id") in ids]
    else:
        selected_records = pick_examples(records, args.pick, args.num_examples)

    if not selected_records:
        raise SystemExit("No records selected for plotting.")

    fig, axes = plt.subplots(len(selected_records), 1, figsize=(8, 3 * len(selected_records)), squeeze=False)
    for idx, rec in enumerate(selected_records):
        steps = rec.get("steps", [])
        traj, stop_t = extract_entropy_trajectory(steps)
        ax = axes[idx][0]
        ax.plot(range(1, len(traj) + 1), traj, marker="o", markersize=3)
        if stop_t is not None:
            ax.axvline(stop_t, color="red", linestyle="--", linewidth=1)
            ax.annotate("stop", xy=(stop_t, traj[stop_t - 1]), xytext=(5, 5),
                        textcoords="offset points", color="red", fontsize=8)
        title = f"{rec.get('example_id')} | n={len(steps)} | correct={rec.get('is_correct')}"
        ax.set_title(title)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Entropy (nats)")
        ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
