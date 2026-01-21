#!/usr/bin/env bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --mem=120G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/fleet_launcher.sh" "$@"
