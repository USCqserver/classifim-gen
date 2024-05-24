#!/bin/python3

import argparse
import datetime
import functools
import resource
import math
import numpy as np
import os
import os.path
import pandas as pd
import scipy.sparse.linalg
import sys
import time
from tqdm import tqdm

ARG_PRESETS = [
    ("thor_defaults",
        "Use default values for the Thor computer",
        {
            "sys_paths": [
                "~/.ipython/",
                "~/d/work/qc4ml/bs_chifc/bitstring-chifc/supplementary_materials/code/"],
            "data_dir0": os.path.expanduser(
                f"/run/media/victor/ssd_T7_Victor/d/qc4ml/bschifc_data/neurips2023"),
            # Just run one (the second) job for testing:
            "num_jobs": 1024,
            "job_id": 1
        })]

def parse_arguments():
    parser = argparse.ArgumentParser(
            prog=f"mnist_*.py",
            description=("Learns on MNIST_CNN dataset."))
    for name, help_str, _ in ARG_PRESETS:
        parser.add_argument(f"--{name}", type=bool, help=help_str)
    parser.add_argument("--sys_paths", type=str, default="",
            help=("Comma-separated list of paths to add to sys.path "
                "(e.g. to bitstring-chifc packages)"))
    parser.add_argument("--data_dir0", type=str, default="",
            help=("Root directory for data files."))
    parser.add_argument("--seed", type=int, default=42,
            help=("Seed to use for rng."))
    parser.add_argument("--num_jobs", type=int, default=1,
            help=("Number of times this script is run in parallel."))
    parser.add_argument("--job_id", type=int, default=0,
            help=("ID of this job."))
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
import classifim_bench.fidelity
import classifim_bench.fft as fft
import classifim_bench.kitaev_hamiltonian as kitaev_hamiltonian
import classifim_bench.gs_cache
import classifim_bench.gs_utils
import classifim_bench.metric
import classifim_bench.plot_tools
import classifim_bench.fft
import classifim_utils

def init_config(args):
    config = {
        "ham_name": "Kitaev20",
        "data_dir0": args.data_dir0}
    config["data_dir"] = os.path.join(
        config["data_dir0"], config["ham_name"].lower())
    config["output_dir"] = os.path.join(
        config["data_dir"], "fermft_out")
    config["seed"] = seed = args.seed
    config["output_file"] = os.path.join(
        config["output_dir"], f"dataset_{seed}_{args.job_id}.npz")
    config["log_file"] = os.path.join(
        config["output_dir"], f"dataset_{seed}_{args.job_id}.log.npz")
    config["mus_cnt"] = mus_cnt = 2 * 10**4
    # Extra value at the beginning for prev_mu:
    all_mus = (np.arange(-1, mus_cnt) + 0.5) / mus_cnt
    # We cover the range [-4, 4]:
    all_mus = -4 + 8 * all_mus
    my_range_start = (args.job_id * mus_cnt) // args.num_jobs
    my_range_end = ((args.job_id + 1) * mus_cnt) // args.num_jobs
    config["param.prev_mu"] = all_mus[my_range_start]
    config["param.mus"] = all_mus[my_range_start + 1:my_range_end + 1]
    return config

def top_evals_rhoA(phi, n=20):
    L = int(0.5 + math.log2(phi.shape[0]))
    assert phi.shape == (1 << L,)
    assert L % 2 == 0
    # Can't request more eigenvalues than size of the density matrix:
    assert n <= (1 << (L // 2))
    rho_a = phi.reshape((2**(L//2), 2**(L//2)))
    # rho_a[i_low, j_low] = sum_{i_high=0,...,2^{L_a}-1}(phi[i_high, i_low] * phi[i_high, j_low]^\dagger):
    rho_a = rho_a.T @ np.conj(rho_a)
    evals = scipy.linalg.eigvalsh(rho_a)
    return evals[:-n-1:-1]

def compute_kitaev_fermft_dataset(config, ham_family):
    prev_mu = config["param.prev_mu"]
    param_mus = config["param.mus"]
    L = ham_family.L
    num_top_evals = 20
    kitaev_dataset = {
        "seed": config["seed"],
        "param.t": 1.0,
        "param.delta": 1.0,
        "param.mus": [],
        "ts": [],
        "deltas": [],
        "samples": [],
        f"top_{num_top_evals}": [],
        "chi_f": []
    }
    prev_gs = ham_family.get_coordinate_basis_ground_state(
        t=kitaev_dataset["param.t"],
        delta=kitaev_dataset["param.delta"],
        mu=prev_mu)

    prng = classifim_utils.DeterministicPrng(config["seed"])

    print(f"{datetime.datetime.now()}: Start generating ({len(param_mus)})",
          flush=True)
    for mu_val in param_mus:
        print(".", end="", flush=True)
        rng = np.random.default_rng(
            prng.get_seed(f"mu={mu_val}"))
        gs = ham_family.get_coordinate_basis_ground_state(
            t=kitaev_dataset["param.t"],
            delta=kitaev_dataset["param.delta"],
            mu=mu_val)
        kitaev_dataset["param.mus"].append(mu_val)
        kitaev_dataset["samples"].append(
                rng.choice(1<<L, size=1400, replace=True, p=np.abs(gs)**2))
        kitaev_dataset[f"top_{num_top_evals}"].append(
            top_evals_rhoA(gs, n=num_top_evals))
        kitaev_dataset["chi_f"].append(np.abs(prev_gs.conj() @ gs))
        prev_gs = gs
    print("", flush=True)

    ds_keys = list(kitaev_dataset.keys())
    for key in ds_keys:
        value = kitaev_dataset[key]
        if isinstance(value, list):
            kitaev_dataset[key] = np.array(value)

    return kitaev_dataset

def main():
    print(f"{datetime.datetime.now()}: Starting main()...")
    t_start = time.time()
    config = init_config(ARGS)
    ham_family = kitaev_hamiltonian.KitaevFamily(L=20)
    kitaev_dataset = compute_kitaev_fermft_dataset(config, ham_family)
    print(
        f"{datetime.datetime.now()}: Saving dataset to {config['output_file']}",
        flush=True)
    np.savez_compressed(config["output_file"], **kitaev_dataset)
    np.savez_compressed(config["log_file"], **config)
    print(f"{datetime.datetime.now()}: Done!")
    t_end = time.time()
    max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Maximum memory used: {max_mem_used} KB")
    print(f"Total time: {t_end - t_start:.3f} seconds")

if __name__ == "__main__":
    main()
