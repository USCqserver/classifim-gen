#!/bin/bash -ue

for experiment_i in 1; do
  (
    set -x
    python3 xxz_opt.py --thor_defaults \
      --experiment_i="${experiment_i}" \
      --num_epochs=200 \
      2>&1 \
    | tee -a "${HOME}/tmp/xxz_opt_${experiment_i}.log"
  )
done
