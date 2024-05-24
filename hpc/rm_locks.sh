#!/bin/bash -ue

# Remove locks left after crashes.
# Verify that no instances of REDACTED are running before running this script.

PROJECT_DIR="REDACTED"
if (( $# >= 2 )); then
  echo "Usage: $0 [RUN_ID]" >&2
  exit 1
fi
ED_OUT_DIR1="${PROJECT_DIR}/ed_out/hubbard_12/lanczos_vec"
ED_OUT_DIR2="${PROJECT_DIR}/ed_out/hubbard_12/lanczos_vec2"
if (( $# == 1 )); then
  if [[ "${1}" == "1" ]]; then
    ED_OUT_DIR="${ED_OUT_DIR1}"
  elif [[ "${1}" == "2" ]]; then
    ED_OUT_DIR="${ED_OUT_DIR2}"
  else
    echo "Usage: $0 [RUN_ID]" >&2
    echo "  RUN_ID must be '1' or '2', not '${1}'." >&2
    exit 1
  fi
else
  ED_OUT_DIR="${ED_OUT_DIR1}"
fi
num_locks_kept=0
num_locks_removed=0
num_stat_failures=0
shopt -s nullglob
for file_name in "${ED_OUT_DIR}"/*.lock; do
  # Last modification time of the file in seconds since Epoch:
  last_modified=$(stat -c %Y "${file_name}" || echo "stat_failure")
  if [[ "${last_modified}" == "stat_failure" ]]; then
    (( num_stat_failures++ )) || true
    continue
  fi
  # Current time in seconds since Epoch:
  current_time=$(date +%s)
  file_age=$(( current_time - last_modified ))
  if (( file_age > 9000 )); then
    echo "rm ${file_name} (${file_age}s)."
    rm "${file_name}"
    (( num_locks_removed++ )) || true
  else
    echo "Fresh ${file_name} (${file_age}s)."
    (( num_locks_kept++ )) || true
  fi
done
if (( num_locks_removed > 0 )); then
  echo "Removed ${num_locks_removed} locks."
fi
if (( num_locks_kept > 0 )); then
  echo "Kept ${num_locks_kept} locks."
fi
if (( num_stat_failures > 0 )); then
  echo "Stat failures for ${num_locks_removed} locks."
fi

npz_files=($(find "${ED_OUT_DIR}/" -type f -name "*.npz" ! -name "*.failure.npz"))
num_npz_files=${#npz_files[@]}
if (( num_npz_files > 0 )); then
  echo "Note: there are ${num_npz_files} .npz files in '${ED_OUT_DIR}'."
fi

failure_npz_files=($(find "${ED_OUT_DIR}/" -type f -name "*.failure.npz"))
num_failure_npz_files=${#failure_npz_files[@]}
if (( num_failure_npz_files > 0 )); then
  echo "Note: there are ${num_failure_npz_files} .failure.npz files in '${ED_OUT_DIR}'."
fi
