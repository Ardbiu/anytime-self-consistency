#!/bin/bash

#!/bin/bash

# =================================================================
# ICML FLEET COMMANDER: 35-GPU L40S SWARM 🐝🐝🐝
# Sharded hero run + non-sharded global run (correct for global budgets).
# =================================================================

mkdir -p outputs/logs

RUN_GROUP=${1:-icml_final}
NUM_SHARDS=${NUM_SHARDS:-35}
PARTITION=mit_preemptable
GPU_TYPE=l40s
MEM=48G
TIME=48:00:00
ENV_NAME=anytime-sc
CFG_SHARDED=configs/paper_hero_sharded.yaml
CFG_GLOBAL=configs/paper_hero_global.yaml

echo "🚀 Launching HERO shards 0-$((NUM_SHARDS - 1)) on ${NUM_SHARDS} ${GPU_TYPE} GPUs..."
JOB_IDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    jid=$(sbatch --partition=${PARTITION} --gres=gpu:${GPU_TYPE}:1 --mem=${MEM} --time=${TIME} --job-name=hero_shard${i} \
      --wrap="module load anaconda3 || module load anaconda/2022.05-x86_64; eval \"\$(conda shell.bash hook)\"; conda activate ${ENV_NAME}; python -m src.run_eval --config ${CFG_SHARDED} --run_group ${RUN_GROUP} --shard_id ${i} --num_shards ${NUM_SHARDS}" \
      --output=outputs/logs/hero_shard${i}.log | awk '{print $4}')
    JOB_IDS+=(${jid})
done

dep=$(IFS=:; echo "${JOB_IDS[*]}")
echo "🚀 Scheduling GLOBAL run after shards complete..."
sbatch --partition=${PARTITION} --gres=gpu:${GPU_TYPE}:1 --mem=${MEM} --time=${TIME} --job-name=hero_global \
  --dependency=afterany:${dep} \
  --wrap="module load anaconda3 || module load anaconda/2022.05-x86_64; eval \"\$(conda shell.bash hook)\"; conda activate ${ENV_NAME}; python -m src.run_eval --config ${CFG_GLOBAL} --run_group ${RUN_GROUP}" \
  --output=outputs/logs/hero_global.log

echo "✅ SWARM DEPLOYED!"
echo "Run group: ${RUN_GROUP}"
echo "Run 'squeue -u adixit1' to monitor."
