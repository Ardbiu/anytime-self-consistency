#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/paper_hero.yaml}"
RUN_GROUP_BASE="${RUN_GROUP_BASE:-icml_fleet_$(date +%m%d%H%M)}"
STOP_FILE="${STOP_FILE:-outputs/STOP_FLEET}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
MEMORY="${MEMORY:-120G}"
DEFAULT_GRES="gpu:h200:1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs}"
RESUBMIT_RETRIES="${RESUBMIT_RETRIES:-30}"
RESUBMIT_DELAY="${RESUBMIT_DELAY:-30}"

# Default to 3 preemptable slots to leave a buffer for resubmission.
DEFAULT_PARTITIONS=("mit_preemptable" "mit_preemptable" "mit_preemptable")

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
  else
    cd "$ROOT_DIR"
  fi

  mkdir -p "$LOG_DIR" "$OUT_DIR"

  PARTITION="${PARTITION:-${SLURM_JOB_PARTITION:-mit_normal_gpu}}"
  GRES="${GRES:-${SLURM_JOB_GRES:-$DEFAULT_GRES}}"
  NUM_SHARDS="${NUM_SHARDS:-4}"
  SHARD_ID="${SHARD_ID:-0}"

  resubmitted=0
  resubmit() {
    if [[ "$resubmitted" -eq 0 && ! -f "$STOP_FILE" ]]; then
      resubmitted=1
      attempt=1
      while :; do
        set +e
        out=$(sbatch -p "$PARTITION" --gres="$GRES" --time="$TIME_LIMIT" --mem="$MEMORY" \
          --chdir="$ROOT_DIR" --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" \
          --export=ALL,CONFIG="$CONFIG",RUN_GROUP_BASE="$RUN_GROUP_BASE",NUM_SHARDS="$NUM_SHARDS",SHARD_ID="$SHARD_ID",STOP_FILE="$STOP_FILE",PARTITION="$PARTITION",GRES="$GRES",TIME_LIMIT="$TIME_LIMIT",MEMORY="$MEMORY",LOG_DIR="$LOG_DIR",OUT_DIR="$OUT_DIR",RESUBMIT_RETRIES="$RESUBMIT_RETRIES",RESUBMIT_DELAY="$RESUBMIT_DELAY" \
          "$0")
        status=$?
        set -e

        if [[ "$status" -eq 0 ]]; then
          echo "[*] Resubmitted shard ${SHARD_ID}: ${out}"
          break
        fi

        echo "[!] Resubmit failed for shard ${SHARD_ID} (attempt ${attempt}/${RESUBMIT_RETRIES}): ${out}"
        if [[ -f "$STOP_FILE" ]]; then
          echo "[*] Stop file detected. Aborting resubmit retries."
          break
        fi
        if [[ "$attempt" -ge "$RESUBMIT_RETRIES" ]]; then
          echo "[!] Giving up resubmit after ${RESUBMIT_RETRIES} attempts."
          break
        fi
        attempt=$((attempt + 1))
        sleep "$RESUBMIT_DELAY"
      done
    fi
  }

  trap 'resubmit; exit 0' TERM INT
  trap 'resubmit' EXIT

  if command -v module >/dev/null 2>&1; then
    module load deprecated-modules || true
    module load gcc/12.2.0-x86_64 || true
    module load python/3.10.8-x86_64 || true
  fi

  if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate anytime-env || true
  fi

  RUN_GROUP="${RUN_GROUP_BASE}_shard${SHARD_ID}"

  set +e
  python -m src.run_eval \
    --config "$CONFIG" \
    --run_group "$RUN_GROUP" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS" \
    --resume \
    --save_interval 10
  status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[!] run_eval exited with status $status; resubmitting shard ${SHARD_ID}."
  fi
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

mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "[*] Launching ${NUM_SHARDS}-slot H200 fleet..."
echo "[*] Run group base: ${RUN_GROUP_BASE}"
echo "[*] Partitions: ${PARTITION_LIST[*]}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  part="${PARTITION_LIST[$shard]}"
  sbatch -p "$part" --gres="$GRES" --time="$TIME_LIMIT" --mem="$MEMORY" \
    --chdir="$ROOT_DIR" --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" \
    --export=ALL,CONFIG="$CONFIG",RUN_GROUP_BASE="$RUN_GROUP_BASE",NUM_SHARDS="$NUM_SHARDS",SHARD_ID="$shard",STOP_FILE="$STOP_FILE",PARTITION="$part",GRES="$GRES",TIME_LIMIT="$TIME_LIMIT",MEMORY="$MEMORY",LOG_DIR="$LOG_DIR",OUT_DIR="$OUT_DIR" \
    "$0"
done

echo "[*] Fleet deployed. Create ${STOP_FILE} to stop auto-resubmit."
