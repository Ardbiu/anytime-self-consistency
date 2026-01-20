#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --mem=120G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

CONFIG="${CONFIG:-configs/paper_hero.yaml}"
RUN_GROUP_BASE="${RUN_GROUP_BASE:-icml_fleet}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_ID="${SHARD_ID:-0}"
STOP_FILE="${STOP_FILE:-outputs/STOP_FLEET}"

mkdir -p logs outputs

if command -v module >/dev/null 2>&1; then
  module load deprecated-modules
  module load gcc/12.2.0-x86_64
  module load python/3.10.8-x86_64
fi

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  source activate anytime-env || true
fi

RUN_GROUP="${RUN_GROUP_BASE}_shard${SHARD_ID}"

python -m src.run_eval \
  --config "$CONFIG" \
  --run_group "$RUN_GROUP" \
  --shard_id "$SHARD_ID" \
  --num_shards "$NUM_SHARDS" \
  --resume \
  --save_interval 10

if [ -f "$STOP_FILE" ]; then
  echo "Stop file detected ($STOP_FILE). Not resubmitting."
  exit 0
fi

sbatch --dependency=afterany:$SLURM_JOB_ID \
  --export=ALL,CONFIG="$CONFIG",RUN_GROUP_BASE="$RUN_GROUP_BASE",NUM_SHARDS="$NUM_SHARDS",SHARD_ID="$SHARD_ID",STOP_FILE="$STOP_FILE" \
  fleet_worker.sh
