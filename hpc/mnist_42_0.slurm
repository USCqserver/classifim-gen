#!/bin/bash
#SBATCH --account=lidar_90
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:28:00
#SBATCH --array=0-9

echo "SLURM_ARRAY_TASK_COUNT: '${SLURM_ARRAY_TASK_COUNT:-UNSET}'"
if [[ -z "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  NUM_JOBS=1
else
  if [[ "${SLURM_ARRAY_TASK_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    NUM_JOBS="${SLURM_ARRAY_TASK_COUNT}"
  else
    NUM_JOBS=1
  fi
fi

if (( NUM_JOBS > 1 )); then
  JOB_ID="${SLURM_ARRAY_TASK_ID}"
else
  JOB_ID=0
fi

module purge
module load conda
source /home1/kasatkin/.local/bin/env_mamba.sh
PROJECT_DIR="/project/lidar_90/kasatkin/bs_chifc"
conda activate "${PROJECT_DIR}/venv/classifim"
echo "$(date -Is): Starting using $(which python3)".

(
  set -x
  python3 "${PROJECT_DIR}/classifim/hpc/mnist_42_0.py" \
    --sys_paths="${PROJECT_DIR}/classifim" \
    --data_dir0="${PROJECT_DIR}/data" \
    --num_jobs="${NUM_JOBS}" \
    --job_id="${JOB_ID}"
)
