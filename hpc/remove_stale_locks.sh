#!/bin/bash -ue

LOCK_FILES=()
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      LOCK_FILES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#LOCK_FILES[@]} == 0 ]]; then
  echo "No lockfiles were provided." >&2
  echo "Usage: $0 [--dry-run] <lockfile> [<lockfile> ...]" >&2
  exit 1
fi

set -- "${LOCK_FILES[@]}"

function read_lockfile_data() {
  local lockfile=$1
  # Empty the associative array:
  declare -gA lockfile_data

  # Read the second line of the file
  local data=""
  local res=0
  data=$(sed -n '2p' "$lockfile") || res=$?
  if [[ $res -ne 0 ]]; then
    return $res
  fi

  # Split the line into key-value pairs
  local IFS=','
  for kv in $data; do
    # Split each key-value pair and store in associative array
    IFS='=' read -r key value <<< "$kv"
    lockfile_data["$key"]="$value"
  done
  return 0
}

is_slurm_job_running() {
  local job_id="$1"
  if squeue --cluster discovery -j "$job_id" -o "%A" 2>/dev/null | grep -q "$job_id"; then
    return 0
  else
    return 1
  fi
}

declare -A lockfile_data

while [[ $# -gt 0 ]]; do
  lockfile="$1"
  shift
  lockfile_name=$(basename "$lockfile")
  res=0
  read_lockfile_data "${lockfile}" || res=$?
  if [[ $res -ne 0 ]]; then
    if [[ -f "$lockfile" ]]; then
      echo "${lockfile_name}: failed to read"
    else
      echo "${lockfile_name}: not found"
    fi
    continue
  fi
  # Now check if SLURM_JOB_ID is set:
  if ! [ "${lockfile_data["SLURM_JOB_ID"]+isset}" ]; then
    echo "${lockfile_name}: SLURM_JOB_ID is not set"
    continue
  fi
  # Now check if the job is still running:
  if is_slurm_job_running "${lockfile_data["SLURM_JOB_ID"]}"; then
    echo "${lockfile_name}: job ${lockfile_data["SLURM_JOB_ID"]} is still running"
    continue
  fi
  # The job is not running, so remove the lockfile:
  if [[ "$DRY_RUN" != false ]]; then
    echo rm "$lockfile"
  else
    (
      set -x
      rm "$lockfile"
    )
  fi
done
