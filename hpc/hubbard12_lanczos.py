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

ARG_PRESETS = [
    ("local_defaults",
        "Use default values for launching this on a local computer",
        {
            "sys_paths": ["~/.ipython/", "~/REDACTED"],
            "output_path": os.path.expanduser(
                "~/REDACTED/hubbard_12/lanczos_vec"),
            "lock_path": os.path.expanduser(
                "~/REDACTED/hubbard_12/lock")
        }),
    ("local_test_defaults",
        "Use default values for testing this on a local computer",
        {
            "sys_paths": ["~/.ipython/", "~/REDACTED"],
            "output_path": os.path.expanduser(
                "~/REDACTED/hubbard_12/lanczos_vec"),
            "lock_path": os.path.expanduser(
                "~/REDACTED/hubbard_12/lock"),
            "max_iters": 1,
            "eigsh_ncv": 5,
            "eigsh_maxiter": 5
        })
 ]

MODEL_NAME = "hubbard_12"

def parse_arguments():
    parser = argparse.ArgumentParser(
            prog="hubbard_gs.py",
            description="Computes the low energy state of the Hubbard Hamiltonian on HPC")
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
    parser.add_argument("--eigsh_ncv", type=int, default=160,
            help=("ncv parameter for scipy.sparse.linalg.eigsh; see "
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html"
            " for more details"))
    parser.add_argument("--eigsh_maxiter", type=int, default=160,
            help=("maxiter parameter for scipy.sparse.linalg.eigsh; see "
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html"
            " for more details"))
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
import classifim_bench.hubbard_hamiltonian
import classifim_bench.gs_cache

from numpy.linalg import norm
def normalize(v):
    return v / norm(v)

def get_lambdas(max_log2_resolution=6, job_id=0, num_jobs=1, random_order=False):
    """Returns a list of (lambda1, lambda2) tuples to be processed by this job.
    """
    res = []
    for log2_resolution in range(max_log2_resolution + 1):
        resolution = 2**log2_resolution
        cur_res = [
            (lambda1i / resolution, lambda2i / resolution)
            for lambda1i in range(resolution)
            for lambda2i in range(resolution)
            if resolution == 1 or lambda1i % 2 == 1 or lambda2i % 2 == 1]
        res.extend(cur_res)
    job_id = job_id % num_jobs
    res = res[job_id::num_jobs]
    if random_order:
        random.shuffle(res)
    return res

def get_params_dicts(hubbard_family, **kwargs):
    """Returns a list of parameter dictionaries to be processed by this job.
    """
    f = classifim_bench.hubbard_hamiltonian.gen_lambdas_to_params_f(
            hubbard_family)
    lambdas = get_lambdas(**kwargs)
    return [f(lam1, lam2) for lam1, lam2 in lambdas]

def debug_print_dict(d, indent=0):
    for key, value in d.items():
        value_type = type(value)
        print(" " * indent, end="", file=sys.stderr)
        print(f"{key}: ", end="", file=sys.stderr)
        if value_type in (int, float, str, bool, type(None)):
            print(f"{value} (type:{value_type.__name__})", file=sys.stderr)
        elif isinstance(value, np.ndarray):
            print(f"np.ndarray(dtype:{value.dtype}, shape:{value.shape})",
                  file=sys.stderr)
        elif isinstance(value, dict):
            print(f"dict", file=sys.stderr)
            debug_print_dict(value, indent + 2)
        elif isinstance(value, tuple) or isinstance(value, list):
            print(f"{value_type.__name__}", file=sys.stderr)
            debug_print_dict(dict(enumerate(value)), indent + 2)
        else:
            print(f"(type:{value_type})", file=sys.stderr)

def _reshape_eigsh_output(vals, vecs, k=None, space_dim=None, is_failure=False):
    vecs = vecs.T
    assert isinstance(vecs, np.ndarray)
    assert isinstance(vals, np.ndarray)
    if k is None:
        k = vecs.shape[0]
    if space_dim is None:
        space_dim = vecs.shape[1]
    if is_failure:
        actual_k = vals.shape[0]
        assert 0 <= actual_k
        assert actual_k <= k
        k = actual_k
    assert vals.shape == (k,)
    assert vecs.shape == (k, space_dim), f"{vecs.shape =} != ({k}, {space_dim})"
    assert vals.dtype == np.float64, f"vals.dtype == {vals.dtype} != np.float64"
    assert vecs.dtype == np.float64, f"vecs.dtype == {vecs.dtype} != np.float64"
    return vals, vecs

def compute_hubbard_12_lanczos(params_dict, ham_family, k, ncv, maxiter, payload):
    t_start = time.time()
    params_vec = tuple(params_dict[key] for key in ham_family.PARAM_NAMES)
    apply_h = functools.partial(ham_family.apply_h, params_vec)
    space_dim = ham_family.space_dim
    apply_h_scipy = scipy.sparse.linalg.LinearOperator(
            shape=(space_dim, space_dim), matvec=apply_h)
    try:
        vals, vecs = scipy.sparse.linalg.eigsh(
                apply_h_scipy, k=k, which='SA', ncv=ncv, maxiter=maxiter)
        vals, vecs = _reshape_eigsh_output(vals, vecs, k, space_dim)
        return {"vals": vals, "vecs": vecs, "version": "0.1.0 (np.float64)",
                "time": time.time() - t_start, **payload}
    except scipy.sparse.linalg.ArpackNoConvergence as e:
        print(f"ArpackNoConvergence: {e}", file=sys.stderr)
        debug_print_dict(
            {key: getattr(e, key) for key in dir(e) if not key.startswith("_")},
            indent=2)
        vals, vecs = _reshape_eigsh_output(
            e.eigenvalues, e.eigenvectors, k, space_dim, is_failure=True)
        failure_dict = {
            "error": "ArpackNoConvergence",
            "args": np.array(e.args),
            "vals": e.eigenvalues,
            "vecs": e.eigenvectors,
            "version": "0.1.0 (np.float64)",
            "time": time.time() - t_start,
            **payload}
        print("failure_dict:", file=sys.stderr)
        debug_print_dict(failure_dict, indent=2)
        raise classifim_bench.gs_cache.GroundStateComputationError(
            "ArpackNoConvergence",
            *e.args,
            failure_dict=failure_dict)

def main():
    if ARGS.soft_time_limit:
        soft_deadline = time.time() + ARGS.soft_time_limit
    else:
        soft_deadline = None
    with classifim_bench.gs_utils.ProcessLockFile(ARGS.lock_path) as process_lock:
        # main processing loop
        hubbard_12_family_float64 = classifim_bench.hubbard_hamiltonian.Hubbard1DFamily(
                dtype_float=np.float64, xp=np)
        param_dicts = get_params_dicts(
                hubbard_12_family_float64,
                job_id=ARGS.job_id,
                num_jobs=ARGS.num_jobs,
                random_order=ARGS.random_order)
        hubbard_12_lanczos_cache = classifim_bench.GroundStateCache(
            compute_ground_state=functools.partial(
                compute_hubbard_12_lanczos,
                ham_family=hubbard_12_family_float64,
                k=ARGS.num_eigsh_evecs,
                ncv=ARGS.eigsh_ncv,
                maxiter=ARGS.eigsh_maxiter,
                payload={
                    "args_keys": np.array(list(ARGS.args_dict_str.keys())),
                    "args_values": np.array(list(ARGS.args_dict_str.values()))
                    }),
            param_keys=("u", "tnn", "tnnn"),
            model_name=MODEL_NAME,
            save_path=ARGS.output_path,
            disable_meta=True)
        num_iters = 0
        num_skipped = 0
        for param_dict in param_dicts:
            if not process_lock.exists():
                print(f"Lock file '{ARGS.lock_path}' removed; exiting.")
                break
            cur_time = time.time()
            if soft_deadline and cur_time > soft_deadline:
                print(f"Soft time limit {ARGS.soft_time_limit} reached "
                    f"({cur_time} > {soft_deadline}); exiting.")
                break
            if hubbard_12_lanczos_cache.cache_exists(param_dict, cache_type="any"):
                print("Skipping", param_dict)
                num_skipped += 1
                continue
            print("Processing", param_dict)
            try:
                res = hubbard_12_lanczos_cache.get_ground_state(
                        param_dict,
                        load=False,
                        verbose=True)
            except classifim_bench.gs_cache.GroundStateComputationError:
                # Set to a truthy value to indicate that the computation was attempted.
                res = 'error'
            except filelock._error.Timeout:
                print(f"Timeout waiting for lock; skipping {param_dict}.")
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
