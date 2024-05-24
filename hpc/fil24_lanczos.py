#!/bin/python3

import argparse
import datetime
import filelock
import functools
import hashlib
import math
import numpy as np
import os
import random
import resource
import scipy.sparse.linalg
import scipy.special
import sys
import time

HAM_NAME = "fil24"

ARG_PRESETS = [
    ("local_defaults",
        "Use default values for launching this on a local computer",
        {
            "sys_paths": ["~/.ipython/", "~/REDACTED"],
            "output_path": os.path.expanduser(
                f"~/REDACTED/{HAM_NAME}/lanczos_vec"),
            "lock_path": os.path.expanduser(
                f"~/REDACTED/{HAM_NAME}/lock")
        }),
    ("local_test_defaults",
        "Use default values for testing this on a local computer",
        {
            "sys_paths": ["~/.ipython/", "~/REDACTED"],
            "output_path": os.path.expanduser(
                f"~/REDACTED/{HAM_NAME}/lanczos_vec"),
            "lock_path": os.path.expanduser(
                f"~/REDACTED/{HAM_NAME}/lock"),
            "max_iters": 1,
            "eigsh_ncv": 20,
            "eigsh_maxiter": 20
        })
 ]

def parse_arguments():
    parser = argparse.ArgumentParser(
            prog=f"{HAM_NAME}_lanczos.py",
            description=(
                "Computes the low energy state of the Frustrated Ising Ladder "
                "on a 24-site lattice on an HPC cluster"))
    for name, help_str, _ in ARG_PRESETS:
        parser.add_argument(f"--{name}", type=bool, help=help_str)
    parser.add_argument("--sys_paths", type=str, default="",
            help=("Comma-separated list of paths to add to sys.path "
                "(e.g. to classifim packages)"))
    parser.add_argument("--output_path", type=str, default="",
            help="Path to the output directory")
    parser.add_argument("--lock_path", type=str, default="",
            help=(
                "Path to the lock file. If provided, the script will\n"
                "  * abort if lock file already exists;\n"
                "  * create lock file;\n"
                "  * stop after current iteration if the lock file is removed;\n"
                "  * remove lock file on exit."))
    parser.add_argument("--max_iters", type=int, default=-1,
            help=("Maximum number of iterations (diagonalizations). "
                "Ignored if negative."))
    parser.add_argument("--soft_time_limit", type=int, default=0,
            help=("Soft time limit in seconds; if positive, the script will not "
                  "start any new diagonalizations after this time"))
    parser.add_argument("--num_jobs", type=int, default=1,
            help="Number of copies of this script running in parallel.")
    parser.add_argument("--job_id", type=int, default=0,
            help="Id of this job among num_jobs parallel jobs.")
    parser.add_argument("--random_order", type=bool, default=False,
            help=("If true, the lambdas are processed in random order.\n"
                "This can be helpful, e.g., when multiple jobs with the same "
                "job_id are running in parallel."))
    parser.add_argument("--num_eigsh_evecs", "--eigsh_k", type=int, default=4,
            help=("Number of eigenvectors to compute with "
                "scipy.sparse.linalg.eigsh (parameter k)"))
    parser.add_argument("--eigsh_ncv", type=int, default=40,
            help=("ncv parameter for scipy.sparse.linalg.eigsh; see "
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html"
            " for more details"))
    parser.add_argument("--eigsh_maxiter", type=int, default=40,
            help=("maxiter parameter for scipy.sparse.linalg.eigsh; see "
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html"
            " for more details"))
    parser.add_argument("--beta", "--inverse_temperature", type=float, default=1.0e7,
            help="Inverse temperature for low-temperature probability distribution.")
    args = parser.parse_args()
    for name, _, default_dict in ARG_PRESETS:
        if getattr(args, name):
            for k, v in default_dict.items():
                if getattr(args, k) == parser.get_default(k):
                    setattr(args, k, v)
    args_dict_str = {k: str(v) for k, v in vars(args).items()}
    if args.sys_paths and not isinstance(args.sys_paths, list):
        args.sys_paths = args.sys_paths.split(",")
    args.args_dict_str = args_dict_str
    return args

ARGS = parse_arguments()

def maybe_add_sys_paths(paths):
    for path in paths:
        if not path:
            continue
        my_pkg_paths = [path, os.path.expanduser(path)]
        my_pkg_paths.append(os.path.realpath(my_pkg_paths[-1]))
        if all(x not in sys.path for x in my_pkg_paths):
            sys.path.append(my_pkg_paths[-1])

maybe_add_sys_paths(ARGS.sys_paths)

import classifim_bench
import classifim_bench.fil24_hamiltonian
import classifim_bench.gs_cache
import classifim_bench.hpc_tools

def get_params_vecs(**kwargs):
    """Returns a list of parameter vectors to be processed by this job.
    """
    f = classifim_bench.fil24_hamiltonian.Fil24ParamConversions.lambdas_to_params
    lambdas = classifim_bench.hpc_tools.get_lambdas(**kwargs)
    return [f(lambda0, lambda1) for lambda0, lambda1 in lambdas]

def main():
    if ARGS.soft_time_limit:
        soft_deadline = time.time() + ARGS.soft_time_limit
    else:
        soft_deadline = None
    with classifim_bench.gs_utils.ProcessLockFile(ARGS.lock_path) as process_lock:
        # main processing loop
        ham_family = classifim_bench.fil24_hamiltonian.CppFil1DFamily(
            nsites=12, edge_dirs=[1, 3, 9, 11])
        params_vecs = get_params_vecs(
                job_id=ARGS.job_id,
                num_jobs=ARGS.num_jobs,
                random_order=ARGS.random_order)
        # Create output directory if it does not exist.
        # Do not create multiple levels of directories since if parent_dir
        # does not exist, it is likely a typo.
        if not os.path.exists(ARGS.output_path):
            parent_dir = os.path.dirname(ARGS.output_path)
            assert os.path.exists(parent_dir), (
                    f"Parent directory '{parent_dir}' does not exist.")
            os.mkdir(ARGS.output_path)
        lanczos_cache = classifim_bench.GroundStateCache(
            compute_ground_state=functools.partial(
                classifim_bench.gs_utils.compute_lanczos,
                ham_family=ham_family,
                k=ARGS.num_eigsh_evecs,
                ncv=ARGS.eigsh_ncv,
                maxiter=ARGS.eigsh_maxiter,
                beta=ARGS.beta,
                payload={
                    "args_keys": np.array(list(ARGS.args_dict_str.keys())),
                    "args_values": np.array(list(ARGS.args_dict_str.values()))
                    }),
            param_keys=ham_family.PARAM_NAMES,
            ham_name=HAM_NAME,
            save_path=ARGS.output_path,
            filename_source=classifim_bench.gs_cache.FS_FILESYSTEM)
        num_iters = 0
        num_skipped = 0
        for params_vec in params_vecs:
            if not process_lock.exists():
                print(f"Lock file '{ARGS.lock_path}' removed; exiting.")
                break
            cur_time = time.time()
            if soft_deadline and cur_time > soft_deadline:
                print(f"Soft time limit {ARGS.soft_time_limit} reached "
                    f"({cur_time} > {soft_deadline}); exiting.")
                break
            if lanczos_cache.cache_exists(params_vec, cache_type="any"):
                print("Skipping", params_vec)
                num_skipped += 1
                continue
            print("Processing", params_vec)
            try:
                res = lanczos_cache.get_ground_state(
                    params_vec,
                    load=False,
                    verbose=True)
            except classifim_bench.gs_cache.GroundStateComputationError:
                # Set to a truthy value to indicate that the computation was attempted.
                res = 'error'
            except filelock._error.Timeout:
                print(f"Timeout waiting for lock; skipping {params_vec}.")
                res = None
            if res is not None:
                num_iters += 1
            else:
                num_skipped += 1
            if ARGS.max_iters > 0 and num_iters >= ARGS.max_iters:
                print(f"Reached {ARGS.max_iters} iterations; exiting.")
                break
    print(f"Done {num_iters=}, {num_skipped=}.")

if __name__ == "__main__":
    main()
