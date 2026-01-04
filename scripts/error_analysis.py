#!/usr/bin/env python
import argparse
import glob
import json
import os
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/runs")
    parser.add_argument("--run_group", type=str)
    parser.add_argument("--latest_group", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--output", type=str, default="outputs/summaries/error_analysis.csv")
    parser.add_argument("--output_subjects", type=str, default="outputs/summaries/error_analysis_subjects.csv")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.dir, "*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found in {args.dir}")

    run_group = args.run_group
    if args.latest_group:
        run_group = get_latest_run_group(files)
        if not run_group:
            raise SystemExit("No run_group found to analyze.")

    selected_files = []
    for f in files:
        rec = read_first_record(f)
        if run_group and rec.get("run_group") != run_group:
            continue
        if args.dataset and rec.get("dataset") != args.dataset:
            continue
        if args.method and rec.get("method") != args.method:
            continue
        selected_files.append(f)

    if not selected_files:
        raise SystemExit("No files match the requested filters.")

    df = pd.concat([pd.read_json(f, lines=True) for f in selected_files], ignore_index=True)
    if df.empty:
        raise SystemExit("No rows found in selected files.")

    summary = df.groupby(["dataset", "method"]).agg(
        total=("is_correct", "count"),
        accuracy=("is_correct", "mean"),
        correct=("is_correct", "sum"),
        avg_tokens=("total_tokens", "mean"),
        parse_errors=("parse_error", "sum"),
    ).reset_index()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Saved error summary to {args.output}")

    if "subject" in df.columns and df["subject"].notna().any():
        subject_summary = df.groupby(["dataset", "method", "subject"]).agg(
            total=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            correct=("is_correct", "sum"),
        ).reset_index()
        subject_summary.to_csv(args.output_subjects, index=False)
        print(f"Saved subject breakdown to {args.output_subjects}")

if __name__ == "__main__":
    main()
