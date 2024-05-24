#!/bin/bash -ue
# SPDX-License-Identifier: GPL-3.0-or-later
#SBATCH --account=lidar_90
#SBATCH --cluster=discovery
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-99
#SBATCH --time=00:59:00
# Duplicate the above for use within this script:
SBATCH_TIME="00:59:00"
START_TIME=$(date +%s)
echo ".slurm started @ $(date -Is)."

compute_soft_deadline() {
  local hours minutes seconds sbatch_seconds margin soft_deadline
  # Parse SBATCH_TIME into hours, minutes, and seconds
  IFS=':' read -r hours minutes seconds <<< "$SBATCH_TIME"

  # Convert SBATCH_TIME to total seconds
  sbatch_seconds=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))

  # Margin in seconds:
  margin=$((8 * 60))

  # Compute soft deadline
  soft_deadline=$((START_TIME + sbatch_seconds - margin))

  echo ${soft_deadline}
}
SOFT_DEADLINE=$(compute_soft_deadline)

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
  python3 "${PROJECT_DIR}/classifim/hpc/dmrgpy_xxz_getmps.py" \
    --sys_paths="${PROJECT_DIR}/classifim,${PROJECT_DIR}/dmrgpy/src" \
    --data_dir0="${PROJECT_DIR}/data" \
    --num_jobs="${NUM_JOBS}" \
    --job_id="${JOB_ID}" \
    --task="refine1" \
    --soft_deadline="${SOFT_DEADLINE}"
)
