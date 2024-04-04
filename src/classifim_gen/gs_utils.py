"""Utilities for ground state computation"""

import classifim_gen.gs_cache
import collections.abc
import datetime
import numpy as np
import os.path
import scipy.sparse.linalg
import sys
import time
from tqdm import tqdm

class ProcessLockFile:
    """Simple context manager for a process lock file.

    Processing should not start if the lock file already exists
    and should finish when the lock file is removed.

    This is a no-op when path is empty or None.
    """
    def __init__(self, path):
        self.path = path
        self.entered = False

    def __enter__(self):
        if not self.path:
            self.entered = True
            return self
        if os.path.exists(self.path):
            raise RuntimeError("Lock file already exists: {}".format(self.path))
        with open(self.path, "w") as f:
            self.entered = True
            f.write(f"Lock file created at {datetime.datetime.now()} "
                f"by pid {os.getpid()}\n")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.entered:
            return
        self.entered = False
        if not self.path:
            return
        try:
            os.remove(self.path)
        except FileNotFoundError:
            print(f"LockFile.__exit__: lock file '{self.path}' "
                  "already removed.", file=sys.stderr)

    def exists(self):
        if not self.path:
            return self.entered
        return os.path.exists(self.path)

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

def get_truncated_boltzmann_p(vals, vecs, beta):
    probs = vecs**2
    probs /= np.sum(probs, axis=1)[:, np.newaxis]
    assert np.all(vals[:-1] <= vals[1:]), f"vals is not sorted!"
    vals = vals - vals[0]
    coefs = np.exp(-beta * vals)
    coefs /= np.sum(coefs)
    p = coefs @ probs
    p /= np.sum(p)
    return p

def postprocess_eigsh_output(vals, vecs, k=None, space_dim=None, is_failure=False, beta=1.0e7):
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
    probs = get_truncated_boltzmann_p(vals, vecs, beta)
    return vals, vecs, probs

COMPUTE_LANCZOS_VERSION = "0.1.1 (np.float64)"

def compute_lanczos(params_vec, ham_family, k=4, ncv=40, maxiter=40,
        beta=1.0e7, payload=None, verbose=True):
    """
    Computes the ground state of the Hamiltonian family using the Lanczos method.

    This function is designed to be usable for `compute_ground_state` argument
    in `GroundStateCache` constructor as follows:
    ```
    ham_family = ... # should have apply_h_fun method and space_dim property.
    cache = GroundStateCache(
        compute_ground_state=lambda params_vec: compute_lanczos(
            params_vec, ham_family),
        ...)
    ```
    """
    t_start = time.time()
    apply_h = ham_family.apply_h_fun(params_vec)
    space_dim = ham_family.space_dim
    apply_h_scipy = scipy.sparse.linalg.LinearOperator(
            shape=(space_dim, space_dim), matvec=apply_h)
    payload = payload or {}
    payload = {
        "compute_lanczos_args_str": f"k={k}, ncv={ncv}, maxiter={maxiter}, beta={beta}",
        "version": COMPUTE_LANCZOS_VERSION,
        **payload}
    try:
        vals, vecs = scipy.sparse.linalg.eigsh(
                apply_h_scipy, k=k, which='SA', ncv=ncv, maxiter=maxiter)
        vals, vecs, probs = postprocess_eigsh_output(vals, vecs, k, space_dim, beta=beta)
        return {"vals": vals, "vecs": vecs, "probs": probs,
                "time": time.time() - t_start, **payload}
    except scipy.sparse.linalg.ArpackNoConvergence as e:
        if verbose:
            print(f"ArpackNoConvergence: {e}", file=sys.stderr)
        debug_print_dict(
            {key: getattr(e, key) for key in dir(e) if not key.startswith("_")},
            indent=2)
        vals, vecs, probs = postprocess_eigsh_output(
            e.eigenvalues, e.eigenvectors, k, space_dim, beta=beta, is_failure=True)
        failure_dict = {
            "error": "ArpackNoConvergence",
            "error_args": np.array(e.args),
            "vals": vals,
            "vecs": vecs,
            "probs": probs,
            "time": time.time() - t_start,
            **payload}
        if verbose:
            print("failure_dict:", file=sys.stderr)
            debug_print_dict(failure_dict, indent=2)
        raise classifim_gen.gs_cache.GroundStateComputationError(
            "ArpackNoConvergence",
            *e.args,
            failure_dict=failure_dict)

def generate_datasets(
        gs_cache, lambdas, params_vecs, vi_to_z, seeds=(42,),
        size_per_lambda=140, cheat=False, payload=None):
    """
    Generate multiple datasets for Bitstring-ChiFc method.

    Args:
        gs_cache: GroundStateCache object to access 'probs' for each params_vec.
        lambdas: List (or np.ndarray) of lambda values.
        params_vecs: List (or np.ndarray) of params_vecs.
            Should have the same length (i.e. shape[0]) as lambdas.
        vi_to_z: np.ndarray of the same length as npz["probs"] for each npz in gs_cache.
        seeds: Iterable of seeds for random number generators.
        size_per_lambda: Number of samples per row of lambdas.
            Integer or a list of integers of the same length as seeds.
        cheat: If True, include the ground truth probability in the dataset.
        payload: dict or a list of dicts of the same length as seeds,
            which will be included in the returned datasets.

    Returns:
        List of datasets.
    """
    seeds = list(seeds)
    def ensure_list(x):
        if not isinstance(x, dict) and (
                isinstance(x, collections.abc.Iterable)):
            x = list(x)
            assert len(x) == len(seeds)
        else:
            if x is None:
                x = {}
            x = [x] * len(seeds)
        assert len(x) == len(seeds)
        return x
    size_per_lambda = ensure_list(size_per_lambda)
    payload = ensure_list(payload)
    assert isinstance(payload[0], dict), f"{payload[0] =}"

    lambdas = np.asarray(lambdas)
    params_vecs = np.asarray(params_vecs)
    assert lambdas.shape[0] == params_vecs.shape[0]
    num_files = lambdas.shape[0]
    assert len(vi_to_z.shape) == 1
    assert vi_to_z.dtype in (np.uint32, np.int32, np.uint64, np.int64),  f"{vi_to_z.dtype}"

    num_datasets = len(seeds)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    all_zs = [[] for _ in range(num_datasets)]
    if cheat:
        all_probs = [[] for _ in range(num_datasets)]
    for (lambda_, params_vec) in tqdm(zip(lambdas, params_vecs), total=lambdas.shape[0]):
        npz = gs_cache.get_ground_state(tuple(params_vec))
        probs = npz["probs"]
        assert probs.shape == (vi_to_z.shape[0],)
        for (i, rng) in enumerate(rngs):
            vis = rng.choice(len(probs), size=size_per_lambda[i], p=probs)
            zs = vi_to_z[vis]
            assert zs.shape == (size_per_lambda[i],)
            all_zs[i].append(zs)
            if cheat:
                all_probs[i].append(probs[vis])
    res = []
    all_lambdas = np.repeat(lambdas, size_per_lambda[i], axis=0)
    for i in range(num_datasets):
        cur_all_zs = np.array(all_zs[i])
        assert cur_all_zs.shape == (num_files, size_per_lambda[i])
        cur_all_zs = cur_all_zs.flatten()
        cur_res = {
            "lambdas": all_lambdas,
            "zs": cur_all_zs,
            "seed": seeds[i],
            "size_per_lambda": size_per_lambda[i],
            **payload[i]}
        if cheat:
            cur_all_probs = np.array(all_probs[i])
            assert cur_all_probs.shape == (num_files, size_per_lambda[i])
            cur_res["probs"] = cur_all_probs.flatten()
        res.append(cur_res)
    return res

