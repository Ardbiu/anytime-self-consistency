#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/paper_hero.yaml}"
RUN_GROUP_BASE="${RUN_GROUP_BASE:-icml_fleet_$(date +%m%d%H%M)}"
STOP_FILE="${STOP_FILE:-outputs/STOP_FLEET}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
MEMORY="${MEMORY:-120G}"
DEFAULT_GRES="gpu:h200:1"

DEFAULT_PARTITIONS=("mit_normal_gpu" "mit_preemptable" "mit_preemptable" "mit_preemptable")

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p logs outputs

  if command -v module >/dev/null 2>&1; then
    module load deprecated-modules
    module load gcc/12.2.0-x86_64
    module load python/3.10.8-x86_64
  fi

  if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate anytime-env || true
  fi

  PARTITION="${PARTITION:-${SLURM_JOB_PARTITION:-mit_normal_gpu}}"
  GRES="${GRES:-${SLURM_JOB_GRES:-$DEFAULT_GRES}}"
  NUM_SHARDS="${NUM_SHARDS:-4}"
  SHARD_ID="${SHARD_ID:-0}"
  RUN_GROUP="${RUN_GROUP_BASE}_shard${SHARD_ID}"

  resubmitted=0
  resubmit() {
    if [[ "$resubmitted" -eq 0 && ! -f "$STOP_FILE" ]]; then
      resubmitted=1
      sbatch -p "$PARTITION" --gres="$GRES" --time="$TIME_LIMIT" --mem="$MEMORY" \
        --export=ALL,CONFIG="$CONFIG",RUN_GROUP_BASE="$RUN_GROUP_BASE",NUM_SHARDS="$NUM_SHARDS",SHARD_ID="$SHARD_ID",STOP_FILE="$STOP_FILE",PARTITION="$PARTITION",GRES="$GRES",TIME_LIMIT="$TIME_LIMIT",MEMORY="$MEMORY" \
        "$0"
    fi
  }

  trap 'resubmit; exit 0' TERM INT

  python -m src.run_eval \
    --config "$CONFIG" \
    --run_group "$RUN_GROUP" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS" \
    --resume \
    --save_interval 10

  resubmit
  exit 0
fi

if [[ -n "${PARTITIONS:-}" ]]; then
  IFS=',' read -r -a PARTITION_LIST <<< "$PARTITIONS"
else
  PARTITION_LIST=("${DEFAULT_PARTITIONS[@]}")
fi

GRES="${GRES:-$DEFAULT_GRES}"

if [[ -z "${NUM_SHARDS:-}" ]]; then
  NUM_SHARDS="${#PARTITION_LIST[@]}"
fi

if [[ "${#PARTITION_LIST[@]}" -lt "$NUM_SHARDS" ]]; then
  last="${PARTITION_LIST[-1]}"
  while [[ "${#PARTITION_LIST[@]}" -lt "$NUM_SHARDS" ]]; do
    PARTITION_LIST+=("$last")
  done
elif [[ "${#PARTITION_LIST[@]}" -gt "$NUM_SHARDS" ]]; then
  PARTITION_LIST=("${PARTITION_LIST[@]:0:$NUM_SHARDS}")
fi

mkdir -p logs outputs

echo "[*] Launching ${NUM_SHARDS}-slot H200 fleet..."
echo "[*] Run group base: ${RUN_GROUP_BASE}"
echo "[*] Partitions: ${PARTITION_LIST[*]}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  part="${PARTITION_LIST[$shard]}"
  sbatch -p "$part" --gres="$GRES" --time="$TIME_LIMIT" --mem="$MEMORY" \
    --export=ALL,CONFIG="$CONFIG",RUN_GROUP_BASE="$RUN_GROUP_BASE",NUM_SHARDS="$NUM_SHARDS",SHARD_ID="$shard",STOP_FILE="$STOP_FILE",PARTITION="$part",GRES="$GRES",TIME_LIMIT="$TIME_LIMIT",MEMORY="$MEMORY" \
    "$0"
done

echo "[*] Fleet deployed. Create ${STOP_FILE} to stop auto-resubmit."
