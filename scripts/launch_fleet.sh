#!/bin/bash

# =================================================================
# ICML FLEET COMMANDER: L40S SWARM 🐝🐝🐝
# Uses 15 Parallel L40S GPUs to bypass H200 queues.
# Capacity: 48GB VRAM (Plenty for Qwen-7B)
# =================================================================

mkdir -p outputs/logs

# === 1. HERO RUN (GSM8K) - 10 SHARDS ===
echo "🚀 Launching HERO Shards 0-9 on L40S..."
for i in {0..9}
do
    sbatch --partition=mit_preemptable --gres=gpu:l40s:1 --mem=48G --time=48:00:00 --job-name=hero_shard${i} \
      --wrap="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate anytime-sc; python -m src.run_eval --config configs/paper_hero.yaml --run_group icml_final --shard_id ${i} --num_shards 10" \
      --output=outputs/logs/hero_shard${i}.log
done

# === 2. HARD MATH - 1 JOB ===
echo "🚀 Launching HARD MATH on L40S..."
sbatch --partition=mit_preemptable --gres=gpu:l40s:1 --mem=48G --time=48:00:00 --job-name=math_run \
  --wrap="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate anytime-sc; python -m src.run_eval --config configs/paper_hard_math.yaml --run_group icml_final" \
  --output=outputs/logs/math.log

# === 3. SUITE RUN - 4 SHARDS ===
echo "🚀 Launching SUITE Shards 0-3 on L40S..."
for i in {0..3}
do
    sbatch --partition=mit_preemptable --gres=gpu:l40s:1 --mem=48G --time=48:00:00 --job-name=suite_shard${i} \
      --wrap="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate anytime-sc; python -m src.run_eval --config configs/paper_hero_suite.yaml --run_group icml_final --shard_id ${i} --num_shards 4" \
      --output=outputs/logs/suite_shard${i}.log
done

echo "✅ ALL 15 SHIPS DEPLOYED!"
echo "Targeting L40S GPUs (No Queue!)."
echo "Run 'squeue -u adixit1' to monitor."
