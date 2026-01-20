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

def parse_csv_list(value: str):
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Base config yaml")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated seed list, e.g. 0,1,2")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated dataset list")
    parser.add_argument("--limit", type=int, help="Override limit for all datasets")
    parser.add_argument("--run_group", type=str, help="Optional run group id")
    parser.add_argument("--checkpoint_examples", type=int, help="Enable checkpoint early stop after N examples")
    parser.add_argument("--checkpoint_degradation", type=float, help="Max degradation vs greedy before stopping (e.g., 0.2)")
    parser.add_argument("--checkpoint_policy", type=str, help="Policy name for greedy checkpoint baseline")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs")
    parser.add_argument("--save_interval", type=int, default=0, help="Save state every N examples")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    datasets = parse_csv_list(args.datasets)
    if not seeds:
        raise ValueError("No seeds provided.")
    if not datasets:
        raise ValueError("No datasets provided.")

    run_group = args.run_group or time.strftime("%Y%m%d-%H%M%S")
    print(f"Run group: {run_group}")

    if args.checkpoint_examples is not None or args.checkpoint_degradation is not None:
        config["checkpoint"] = {
            "enabled": True,
            "examples": int(args.checkpoint_examples or 50),
            "max_degradation": float(args.checkpoint_degradation or 0.2),
        }
        if args.checkpoint_policy:
            config["checkpoint"]["policy"] = args.checkpoint_policy

    dataset_limits = config.get("dataset_limits", {})
    base_limit = config.get("limit", None)

    for dataset in datasets:
        limit = args.limit if args.limit is not None else dataset_limits.get(dataset, base_limit)
        for seed in seeds:
            print(f"Running dataset={dataset} seed={seed} limit={limit}")
            run_eval(
                config=config,
                dataset_override=dataset,
                limit_override=limit,
                seed_override=seed,
                run_group=run_group,
                resume=args.resume,
                save_interval=args.save_interval,
            )

if __name__ == "__main__":
    main()
