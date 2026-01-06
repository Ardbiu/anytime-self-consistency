#!/usr/bin/env python
import argparse
import time
import yaml
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.run_eval import run_eval

def parse_csv_floats(value: str):
    if value is None:
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]

def parse_csv_ints(value: str):
    if value is None:
        return None
    vals = [int(v.strip()) for v in value.split(",") if v.strip()]
    return vals if vals else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Base config yaml")
    parser.add_argument("--ucb_c", type=str, default="0.5,1.0,2.0", help="Comma-separated ucb_c values")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name for tuning")
    parser.add_argument("--limit", type=int, default=200, help="Train split limit for tuning")
    parser.add_argument("--seed", type=int, default=42, help="Seed for tuning runs")
    parser.add_argument("--global_budgets", type=str, help="Override global budgets, comma-separated ints")
    parser.add_argument("--run_group", type=str, help="Optional run_group id")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    ucb_vals = parse_csv_floats(args.ucb_c)
    if not ucb_vals:
        raise ValueError("No ucb_c values provided.")

    budgets = parse_csv_ints(args.global_budgets)
    run_group = args.run_group or f"ucb_tune_{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Run group: {run_group}")

    for ucb_c in ucb_vals:
        cfg = dict(base_config)
        cfg["dataset"] = args.dataset
        cfg["split"] = "train"
        cfg["limit"] = args.limit
        cfg["seed"] = args.seed

        methods = []
        for method_cfg in base_config.get("methods", []):
            if method_cfg.get("name") != "global_anytime_sc":
                continue
            m = dict(method_cfg)
            m["ucb_c"] = ucb_c
            if budgets is not None:
                m["global_budget_tokens"] = budgets
            methods.append(m)

        if not methods:
            raise ValueError("Config has no global_anytime_sc method to tune.")
        cfg["methods"] = methods

        print(f"Tuning ucb_c={ucb_c} on {args.dataset}[train] limit={args.limit}")
        run_eval(
            config=cfg,
            dataset_override=args.dataset,
            limit_override=args.limit,
            seed_override=args.seed,
            run_group=run_group,
        )

if __name__ == "__main__":
    main()
