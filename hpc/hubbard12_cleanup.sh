#!/bin/bash -ue

INCORRECT_FILES=( hubbard12_1mkg2ms7l1mwz hubbard12_10m5yy0fy04wu hubbard12_07l9pqsx0sbh7 )
PROJECT_DIR="REDACTED"
LANCZOS_PROBS_DIR="${PROJECT_DIR}/REDACTED/hubbard12/lanczos_probs"

for file in "${INCORRECT_FILES[@]}"; do
  rm "${LANCZOS_PROBS_DIR}/$file.npz"
done
