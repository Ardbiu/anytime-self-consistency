#!/usr/bin/env bash
set -euo pipefail

# Manual slot launcher: run the same script in multiple terminals with different SHARD_ID/PARTITION.
# It only submits a job if the shard is not already running or pending.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs}"

CONFIG="${CONFIG:-configs/suite_paper.yaml}"
NUM_SHARDS="${NUM_SHARDS:-4}"
PARTITION="${PARTITION:-mit_preemptable}"
GRES="${GRES:-gpu:h200:1}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
MEMORY="${MEMORY:-120G}"
WATCH_INTERVAL="${WATCH_INTERVAL:-}"

RUN_GROUP_FILE="${RUN_GROUP_FILE:-${OUT_DIR}/fleet_run_group.txt}"

if [[ -z "${SHARD_ID:-}" ]]; then
  echo "[!] SHARD_ID is required. Example: SHARD_ID=0 PARTITION=mit_normal_gpu bash scripts/fleet_slot.sh"
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"

if [[ -n "${RUN_GROUP_BASE:-}" ]]; then
  echo "$RUN_GROUP_BASE" > "$RUN_GROUP_FILE"
elif [[ -f "$RUN_GROUP_FILE" ]]; then
  RUN_GROUP_BASE="$(cat "$RUN_GROUP_FILE")"
else
  RUN_GROUP_BASE="icml_fleet_$(date +%m%d%H%M)"
  echo "$RUN_GROUP_BASE" > "$RUN_GROUP_FILE"
fi

safe_group="${RUN_GROUP_BASE//[^A-Za-z0-9_]/_}"
job_name="fleet_slot_s${SHARD_ID}_${safe_group:0:24}"

submit_slot() {
  if squeue -h -u "$USER" -o "%j" | grep -Fq "$job_name"; then
    echo "[*] Shard ${SHARD_ID} already running or pending as ${job_name}."
    return 0
  fi

  sbatch -p "$PARTITION" --gres="$GRES" --time="$TIME_LIMIT" --mem="$MEMORY" \
    --job-name "$job_name" \
    --chdir="$ROOT_DIR" --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$ROOT_DIR"

if command -v module >/dev/null 2>&1; then
  module load deprecated-modules || true
  module load gcc/12.2.0-x86_64 || true
  module load python/3.10.8-x86_64 || true
fi

if [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate anytime-env || true
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[!] Python not found after env setup."
  exit 1
fi

python -m src.run_eval \
  --config "$CONFIG" \
  --run_group "${RUN_GROUP_BASE}_shard${SHARD_ID}" \
  --shard_id "$SHARD_ID" \
  --num_shards "$NUM_SHARDS" \
  --resume \
  --save_interval 10
EOF
}

if [[ -n "$WATCH_INTERVAL" ]]; then
  while true; do
    submit_slot
    sleep "$WATCH_INTERVAL"
  done
else
  submit_slot
fi
