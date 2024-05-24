#!/bin/python3

import argparse
import glob
import numpy as np
import os
import sys
import time
import hashlib

HAM_NAME = "hubbard12"
PROG_NAME = "hubbard12_verify_gs"

def maybe_split_on_comma(arg):
    """
    If the arg is not a list already, splits it on ','.
    """
    if isinstance(arg, list):
        return arg
    return arg.split(",")

def parse_arguments():
    parser = argparse.ArgumentParser(
            prog=f"{PROG_NAME}.py",
            description="Computes the low energy state of the Hubbard Hamiltonian on HPC")
    parser.add_argument("--sys_paths", type=str, default="",
            help=("Comma-separated list of paths to add to sys.path "
                "(e.g. to classifim packages)"))
    parser.add_argument("--num_jobs", type=int, default=1,
            help="Number of copies of this script running in parallel.")
    parser.add_argument("--job_id", type=int, default=0,
            help="Id of this job among num_jobs parallel jobs.")
    parser.add_argument("--input_paths", type=str, default="",
            help="Comma-separated list of paths to the input directory")
    parser.add_argument("--output_path", type=str, default="",
            help="Path to the output directory")
    parser.add_argument("--beta", "--inverse_temperature", type=float, default=1.0e7,
            help="Inverse temperature for low-temperature probability distribution.")
    parser.add_argument("--max_iters", type=int, default=-1,
            help=("Maximum number of parameter points to process. "
                "Ignored if negative."))
    args = parser.parse_args()
    LIST_ARGS = ["sys_paths", "input_paths"]
    for key in LIST_ARGS:
        setattr(args, key, maybe_split_on_comma(getattr(args, key)))
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

def deterministic_hash(name):
    md5_hash = hashlib.md5()
    md5_hash.update(name.encode('utf-8'))
    return int(md5_hash.hexdigest(), 16)

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

def process_one_parameter_set(log_file, name, input_file_list, output_file, beta):
    all_failures = all(
        f is None or f.endswith('.failure.npz') for f in input_file_list)
    p_dict = {}
    p_sum = 0
    p_cnt = 0
    example_npz = None
    infidelities = []
    for i, input_file in enumerate(input_file_list):
        if input_file is None:
            continue
        is_failure = input_file.endswith('.failure.npz')
        npz = np.load(input_file)
        if 'vals' in npz:
            vals = npz['vals']
            vecs = npz['vecs']
        else:
            vals, vecs = _reshape_eigsh_output(
                npz['eigenvalues'],
                npz['eigenvectors'])
        cur_p = get_truncated_boltzmann_p(vals, vecs, beta)
        if all_failures or not is_failure:
            p_sum = p_sum + cur_p
            p_cnt += 1
            example_npz = npz
        for j, pj in p_dict.items():
            fidelity = np.sum((pj * cur_p)**0.5)
            infidelity = 1 - fidelity
            infidelities.append(infidelity)
            if infidelity < -1e-15:
                log_file.write(
                    f"'{name}: infidelity = {infidelity} < -1e-15 "
                    f"between #{j} and #{i}.\n")
            if infidelity > 1e-7:
                log_file.write(
                    f"'{name}: infidelity = {infidelity} > 1e-7 "
                    f"between #{j} and #{i}.\n")

        p_dict[i] = cur_p
    res = {
        key: example_npz[key]
        for key in example_npz.keys()
        if key not in ["vecs", "eigenvectors", "args_keys", "args_values"]}
    res["probs"] = p_sum / p_cnt
    res["infidelities"] = np.array(infidelities, dtype=np.float64)
    np.savez_compressed(output_file, **res)

def main_loop(
        log_file, input_paths, output_path, max_iters,
        job_id=0, num_jobs=1, **kwargs):
    """
    Process files in directories input_paths.

    For each of these directories we look into files with file
    names of the form <name>.npz or <name>.failure.npz.
    If both <name>.npz and <name>.failure.npz is present in
    the same directory for the same <name>, then <name>.failure.npz is ignored.

    For each <name> the expected situation is that <name>.npz is
    present in each of the directories. Then we should call
    `process_one_parameter_set(input_file_list, output_file, **kwargs)`
    (where output_file is of the form "{output_path}/{name}.npz")

    Otherwise, a message should be logged indicating which files
    are missing for <name> and which are failures.
    process_one_parameter_set(input_file_list, output_file, **kwargs)
    should still be called with None in places of missing files
    and <name>.failure.npz for directories containing only <name>.failure.npz
    and not <name>.npz.
    """
    time_0 = time.time()
    assert len(input_paths) >= 1
    log_file.write("main_loop()\n")
    for i, input_path in enumerate(input_paths):
        log_file.write(f"input#{i}: '{input_path}'\n")
    log_file.write(f"output: '{output_path}'\n")

    input_files = {}
    for i, input_path in enumerate(input_paths):
        for file_path in glob.glob(os.path.join(input_path, '*.npz')):
            base_name = os.path.basename(file_path)
            name1, ext1 = os.path.splitext(base_name)
            assert ext1 == '.npz'
            name2, ext2 = os.path.splitext(name1)
            if ext2 == '.failure':
                name = name2
                is_failure = True
            else:
                name = name1
                is_failure = False
            if name not in input_files:
                input_files[name] = [{} for _ in range(len(input_paths))]
            cur_path_info = input_files[name][i]
            if cur_path_info.get('is_failure', True):
                cur_path_info['is_failure'] = is_failure
                cur_path_info['name'] = name
                cur_path_info['path'] = file_path

    time_1 = time.time()
    log_file.write(f"glob time: {time_1 - time_0:.3f}s\n")

    num_iters = 0
    for name, path_info_lst in input_files.items():
        if num_jobs > 1 and deterministic_hash(name) % num_jobs != job_id:
            continue
        output_file = os.path.join(output_path, f"{name}.npz")
        if os.path.exists(output_file):
            continue
        log_str = []
        input_file_list = []
        for i, path_info in enumerate(path_info_lst):
            is_failure = path_info.get('is_failure', None)
            if is_failure is None:
                log_str.append(f"input#{i} is missing")
                input_file_list.append(None)
            else:
                input_file_list.append(path_info.get('path'))
                if is_failure:
                    log_str.append(f"input#{i} is failure")
        if len(log_str) > 0:
            log_file.write(f"Incomplete inputs for '{name}': "
                           + ", ".join(log_str) + "\n")
        process_one_parameter_set(
                log_file=log_file,
                name=name,
                input_file_list=input_file_list,
                output_file=output_file,
                **kwargs)
        num_iters += 1
        if max_iters > 0 and num_iters >= max_iters:
            print(f"Reached {ARGS.max_iters} iterations; exiting.")
            break
    time_2 = time.time()
    log_file.write(f"processing time: {time_2 - time_1:.3f}s for {num_iters} iters\n")

def main():
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
        log_file_name = f"hubbard12_verify_gs.{job_id}.log"
    else:
        log_file_name = "hubbard12_verify_gs.log"
    log_file_name = os.path.join(ARGS.output_path, log_file_name)
    with open(log_file_name, 'a') as log_file:
        main_loop(
                log_file, ARGS.input_paths, ARGS.output_path,
                ARGS.max_iters, num_jobs=ARGS.num_jobs, job_id=ARGS.job_id,
                beta=ARGS.beta)

if __name__ == "__main__":
    main()
