#!/bin/bash -ue

REDACTED
source REDACTED.sh
PROJECT_DIR="REDACTED"
conda activate "${PROJECT_DIR}/venv/classifim"
LANCZOS_DIR1="${PROJECT_DIR}/ed_out/hubbard_12/lanczos_vec"
NUM_JOBS=REDACTED
JOB_ID=REDACTED

(
  set -x
  python3 "${PROJECT_DIR}/code/hpc/hubbard_2_verify_gs.py" \
    --sys_paths="${PROJECT_DIR}/code" \
    --num_jobs="${NUM_JOBS}" \
    --job_id="${JOB_ID}" \
    --input_paths="${LANCZOS_DIR1},${LANCZOS_DIR1}2" \
    --output_path="${PROJECT_DIR}/ed_out/hubbard_12/lanczos_probs" \
    --max_iters=550
)
