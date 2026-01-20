#!/bin/bash
set -euo pipefail

# Configuration
CONFIG="configs/paper_hero.yaml"
USER_NAME="adixit1"

mkdir -p logs

submit_job() {
    local partition=$1
    local gres=$2
    local time_limit=$3

    local job_id=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH -p $partition
#SBATCH --gres=$gres
#SBATCH --time=$time_limit
#SBATCH --mem=120G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  source activate anytime-env
fi

python -m src.run_eval --config $CONFIG --resume --save_interval 10
EOT
)
    echo "$job_id"
}

echo "[*] Launching ICML Sprint Fleet..."

NORMAL_1=$(submit_job "mit_normal_gpu" "gpu:h200:1" "06:00:00")
sbatch --dependency=afterany:$NORMAL_1 fleet_manager.sh

NORMAL_2=$(submit_job "mit_normal_gpu" "gpu:h200:1" "06:00:00")
sbatch --dependency=afterany:$NORMAL_2 fleet_manager.sh

PREEMPT_1=$(submit_job "mit_preemptable" "gpu:h100:1" "48:00:00")
PREEMPT_2=$(submit_job "mit_preemptable" "gpu:h100:1" "48:00:00")

echo "[*] Fleet deployed."
echo "Normal IDs: $NORMAL_1, $NORMAL_2"
echo "Preemptable IDs: $PREEMPT_1, $PREEMPT_2"
echo "[!] You can now log out. The chain-dependency will keep the GPUs warm."
