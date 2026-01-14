#!/bin/bash
#SBATCH --job-name=anytime-sc-hero
#SBATCH --output=outputs/logs/slurm_%j.log
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu  # Check if 'engaging' or specific partition needed
#SBATCH --constraint=high-capacity  # Optional, verify with cluster docs

# Load modules (Adjust based on cluster specifics)
# module load python/3.10 cuda/12.1

# Activate environment
source venv/bin/activate

echo "Starting job on $(hostname)"
echo "Date: $(date)"

# Install deps if needed (usually done once interactively, but safety check)
pip install -r requirements.txt --quiet

# Run the hero suite
# Adjust config and seed as needed
python -m src.run_eval --config configs/paper_hero.yaml --seed 0 --run_group icml_hero_v1

echo "Job finished at $(date)"
