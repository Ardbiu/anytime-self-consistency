#!/bin/bash

# =================================================================
# ICML FLEET COMMANDER: 35-GPU L40S SWARM 🐝🐝🐝
# MASSIVE PARALLELISM: 35 GPUs dedicated to ONE task.
# Target Runtime: < 2 Hours
# =================================================================

mkdir -p outputs/logs

echo "🚀 Launching HERO Shards 0-34 on 35 L40S GPUs..."

# Launch 35 separate jobs
for i in {0..34}
do
    sbatch --partition=mit_preemptable --gres=gpu:l40s:1 --mem=48G --time=48:00:00 --job-name=hero_shard${i} \
      --wrap="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate anytime-sc; python -m src.run_eval --config configs/paper_hero.yaml --run_group icml_final --shard_id ${i} --num_shards 35" \
      --output=outputs/logs/hero_shard${i}.log
done

echo "✅ SWARM DEPLOYED! (35 Jobs)"
echo "Est. completion time: ~1.5 - 2 hours."
echo "Run 'squeue -u adixit1' to monitor the swarm."
