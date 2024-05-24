#!/bin/bash

REDACTED

source REDACTED.sh
PROJECT_DIR="REDACTED"
conda activate "${PROJECT_DIR}/venv/classifim"
NUM_JOBS=REDACTED
JOB_ID=REDACTED

(
  set -x
  python3 "${PROJECT_DIR}/code/hpc/hubbard12.py" \
    --sys_paths="${PROJECT_DIR}/code" \
    --output_path="${PROJECT_DIR}/ed_out/hubbard12/lanczos_vec2" \
    --soft_time_limit=9000 \
    --num_jobs="${NUM_JOBS}" \
    --job_id="${JOB_ID}" \
    --random_order=True \
    --eigsh_k=4 \
    --eigsh_ncv=140 \
    --eigsh_maxiter=140
)
