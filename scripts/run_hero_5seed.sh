#!/usr/bin/env bash
set -euo pipefail

python scripts/run_suite.py --config configs/paper_hero_suite.yaml --seeds 0,1,2,3,4 --datasets gsm8k,gsm_plus
python scripts/aggregate_results.py --latest_group --bootstrap 1000
