#!/bin/bash
set -euo pipefail

CONFIG="${CONFIG:-configs/paper_hero.yaml}"
NUM_SHARDS="${NUM_SHARDS:-4}"
RUN_GROUP_BASE="${RUN_GROUP_BASE:-icml_fleet_$(date +%m%d%H%M)}"
STOP_FILE="${STOP_FILE:-outputs/STOP_FLEET}"

mkdir -p logs outputs

echo "[*] Launching ${NUM_SHARDS}-slot H200 fleet..."
echo "[*] Run group base: ${RUN_GROUP_BASE}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  sbatch --export=ALL,CONFIG="$CONFIG",RUN_GROUP_BASE="$RUN_GROUP_BASE",NUM_SHARDS="$NUM_SHARDS",SHARD_ID="$shard",STOP_FILE="$STOP_FILE" \
    fleet_worker.sh
done

echo "[*] Fleet deployed. Create ${STOP_FILE} to stop auto-resubmit."
