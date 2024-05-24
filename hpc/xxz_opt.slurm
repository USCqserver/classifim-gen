#!/bin/bash -ue
# SPDX-License-Identifier: GPL-3.0-or-later
#SBATCH --account=lidar_90
#SBATCH --cluster=discovery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-0
#SBATCH --time=00:41:00

module purge
module load conda
source /home1/kasatkin/.local/bin/env_mamba.sh
PROJECT_DIR="/project/lidar_90/kasatkin/bs_chifc"
conda activate "${PROJECT_DIR}/venv/classifim"
echo "$(date -Is): Starting using $(which python3)".

(
  set -x
  python3 "${PROJECT_DIR}/classifim/hpc/xxz_opt.py" \
    --sys_paths="${PROJECT_DIR}/classifim" \
    --data_dir0="${PROJECT_DIR}/data" \
    --experiment_i="${SLURM_ARRAY_TASK_ID}" \
    --num_epochs=400
)
