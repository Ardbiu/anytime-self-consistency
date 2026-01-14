#!/bin/bash

# =================================================================
# ICML FLEET COMMANDER: MAX PRIORITY HERO RUN 🚀🚀🚀
# Dedicates ALL 10 GPUs to the single "Hero" experiment.
# =================================================================

mkdir -p outputs/logs

echo "🚀 Launching HERO Shards 0-5 on H200s..."
# First 6 shards on H200s
for i in {0..5}
do
    sbatch --partition=mit_preemptable --gres=gpu:h200:1 --mem=48G --time=48:00:00 --job-name=hero_shard${i} \
      --wrap="module load anaconda3/2023.07; source activate anytime-sc; python -m src.run_eval --config configs/paper_hero.yaml --run_group icml_final --shard_id ${i} --num_shards 10" \
      --output=outputs/logs/hero_shard${i}.log
done

echo "🚀 Launching HERO Shards 6-9 on H100s..."
# Next 4 shards on H100s
for i in {6..9}
do
    sbatch --partition=mit_preemptable --gres=gpu:h100:1 --mem=48G --time=48:00:00 --job-name=hero_shard${i} \
      --wrap="module load anaconda3/2023.07; source activate anytime-sc; python -m src.run_eval --config configs/paper_hero.yaml --run_group icml_final --shard_id ${i} --num_shards 10" \
      --output=outputs/logs/hero_shard${i}.log
done

echo "✅ ALL 10 SHARDS DEPLOYED FOR HERO RUN!"
echo "Est. completion time: ~5-6 hours."
echo "Run 'squeue -u adixit1' to monitor."
