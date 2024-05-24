#!/bin/bash
#SBATCH --account=lidar_90
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00

module purge
module load conda
source /home1/kasatkin/.local/bin/env_mamba.sh
PROJECT_DIR="/project/lidar_90/kasatkin/bs_chifc"
# mamba env create \
#   --file "${PROJECT_DIR}/bitstring-chifc/carc_scripts/bs_chifc.yml" \
#   --prefix "${PROJECT_DIR}/venv/bs_chifc"
mamba activate "${PROJECT_DIR}/venv/bs_chifc"
mamba install filelock
