#!/bin/bash
set -euo pipefail

# =================================================================
# Single H200 launch: run per-example methods, then global budgets.
# =================================================================

RUN_GROUP=${1:-icml_h200}
PARTITION=${PARTITION:-mit_normal_gpu}
GPU_TYPE=${GPU_TYPE:-h200}
MEM=${MEM:-64G}
TIME=${TIME:-4:00:00}
ENV_NAME=${ENV_NAME:-anytime-sc}
CFG_SHARDED=${CFG_SHARDED:-configs/paper_hero_sharded.yaml}
CFG_GLOBAL=${CFG_GLOBAL:-configs/paper_hero_global.yaml}

mkdir -p outputs/logs

sbatch --partition=${PARTITION} --gres=gpu:${GPU_TYPE}:1 --mem=${MEM} --time=${TIME} --job-name=hero_h200 \
  --wrap="module load deprecated-modules || true; \
    module load anaconda3/2022.05-x86_64 || module load anaconda3 || module load anaconda/2022.05-x86_64 || module load miniconda; \
    eval \"\$(conda shell.bash hook)\"; \
    conda activate ${ENV_NAME}; \
    python -m src.run_eval --config ${CFG_SHARDED} --run_group ${RUN_GROUP}; \
    python -m src.run_eval --config ${CFG_GLOBAL} --run_group ${RUN_GROUP}" \
  --output=outputs/logs/hero_h200.log

echo "Submitted single H200 job (run_group=${RUN_GROUP})."
echo "Check: squeue -u ${USER}"
