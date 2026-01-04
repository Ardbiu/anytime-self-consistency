#!/usr/bin/env python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.run_eval import run_eval

def main():
    config = {
        "dataset": "gsm8k",
        "split": "test",
        "limit": 2,
        "seed": 0,
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_new_tokens": 64,
        "methods": [
            {"name": "greedy"},
        ],
        "output_dir": "outputs/runs/",
    }
    run_eval(config=config, run_group="regression_greedy", seed_override=0)
    print("greedy regression completed")

if __name__ == "__main__":
    main()
