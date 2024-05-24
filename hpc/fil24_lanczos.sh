#!/bin/bash -ue
REDACTED
source REDACTED.sh
PROJECT_DIR="REDACTED"
conda activate "${PROJECT_DIR}/venv/classifim"
NUM_JOBS=REDACTED
JOB_ID=REDACTED

(
  set -x
  python3 "${PROJECT_DIR}/code/hpc/fil24_lanczos.py" \
    --sys_paths="${PROJECT_DIR}/code" \
    --output_path="${PROJECT_DIR}/ed_out/fil24_lanczos/lanczos_vec" \
    --soft_time_limit=2880 \
    --num_jobs="${NUM_JOBS}" \
    --job_id="${JOB_ID}" \
    --random_order=True \
    --eigsh_k=4 \
    --eigsh_ncv=40 \
    --eigsh_maxiter=40
)
