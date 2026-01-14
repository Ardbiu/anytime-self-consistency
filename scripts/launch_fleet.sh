#!/bin/bash

# =================================================================
# ICML FLEET COMMANDER 🚀
# Submits 10 Jobs across H200s and H100s for max parallelism.
# =================================================================

# Create logs directory
mkdir -p outputs/logs

echo "🚀 Launching 5 Shards for HERO RUN (GSM8K) on H200s..."
# Using 5 H200s for the Hero Run (Priority #1)
for i in {0..4}
do
    sbatch --partition=mit_preemptable --gres=gpu:h200:1 --mem=48G --time=48:00:00 --job-name=hero_shard${i} \
      --wrap="python -m src.run_eval --config configs/paper_hero.yaml --run_group icml_final --shard_id ${i} --num_shards 5" \
      --output=outputs/logs/hero_shard${i}.log
done

echo "🚀 Launching 1 Job for HARD MATH on H200..."
# Using 1 H200 for Hard Math (Smaller dataset, hard tokens)
sbatch --partition=mit_preemptable --gres=gpu:h200:1 --mem=48G --time=48:00:00 --job-name=math_run \
  --wrap="python -m src.run_eval --config configs/paper_hard_math.yaml --run_group icml_final" \
  --output=outputs/logs/math.log


echo "🚀 Launching 4 Shards for SUITE RUN (Generalization) on H100s..."
# Using 4 H100s for the massive Suite (GSM8K + PLUS + MATH)
for i in {0..3}
do
    sbatch --partition=mit_preemptable --gres=gpu:h100:1 --mem=48G --time=48:00:00 --job-name=suite_shard${i} \
      --wrap="python -m src.run_eval --config configs/paper_hero_suite.yaml --run_group icml_final --shard_id ${i} --num_shards 4" \
      --output=outputs/logs/suite_shard${i}.log
done

echo "✅ FLEET DEPLOYED!"
echo "Run 'squeue -u adixit1' to monitor your 10 ships."
