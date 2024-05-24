#!/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import argparse
import time
import datetime
import multiprocessing

SUPPORTED_TASKS = ["dmrg", "refine1", "sample", "sample*"]
ARG_PRESETS = [
    ("thor_defaults",
        "Use default values for the Thor computer",
        {
            "sys_paths": [
                "~/.ipython/",
                "~/d/work/qc4ml/bs_chifc/classifim/src/",
                "~/d/projects/classifim/dmrgpy/src/"],
            "data_dir0": os.path.expanduser(
                "/media/victor/ssd_T7_Victor/d/qc4ml/bschifc_data/neurips2023"),
            # Just run one (the second) job for testing:
            "num_jobs": 1024,
            "job_id": 1,
            "task": "sample*",
            # It will likely stop after the first parameter value.
            "soft_deadline": time.time() + 180 # 3 minutes
        })]

def parse_arguments():
    parser = argparse.ArgumentParser(
            prog=f"dmrgpy_xxz_getmps.py",
            description=("Generate XXZ MPSs for 64x64 grid of points."))
    for name, help_str, _ in ARG_PRESETS:
        parser.add_argument(f"--{name}", type=bool, help=help_str)
    parser.add_argument("--sys_paths", type=str, default="",
            help=("Comma-separated list of paths to add to sys.path "
                "(e.g. to bitstring-chifc packages)"))
    parser.add_argument("--data_dir0", type=str, default="",
            help="Root directory for data files.")
    parser.add_argument("--num_jobs", type=int, default=1,
            help="Number of times this script is run in parallel.")
    parser.add_argument("--job_id", type=int, default=0,
            help="ID of this job.")
    parser.add_argument("--soft_deadline", type=int, default=0,
            help="Soft deadline (epoch time in seconds).")
    parser.add_argument("--julia_setup_strategy", type=str, default="",
            help=("One of 'resolve', 'instantiate', 'import';"
                  "'import' is the default and the only option safe to use "
                  "when multiple instances are running in parallel."))
    parser.add_argument("--task", type=str, default="",
            help="Task to complete: " + " or ".join(SUPPORTED_TASKS))
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

import dmrgpy
import dmrgpy.juliarun
import numpy as np
import classifim_utils
import functools
import socket

if ARGS.julia_setup_strategy:
    dmrgpy.juliarun.get_jlsession(ARGS.julia_setup_strategy)

LAMBDA0I_MAX = 63
LAMBDA1I_MAX = 63

def get_num_cpu_cores(verbose=False):
    cores = os.environ.get("SLURM_CPUS_PER_TASK")
    if cores:
        if verbose:
            print(f"SLURM_CPUS_PER_TASK={cores}", file=sys.stderr)
        return int(cores)
    cores = multiprocessing.cpu_count()
    if verbose:
        print(f"multiprocessing.cpu_count()={cores}", file=sys.stderr)
    return cores

def init_config(args):
    config = {
        "ham_name": "XXZ1d",
        "n_sites": 300}
    for key in ["data_dir0", "num_jobs", "job_id", "soft_deadline"]:
        config[key] = getattr(args, key)
    config["data_dir"] = os.path.join(
        config["data_dir0"], config["ham_name"].lower())
    task = args.task
    assert task in SUPPORTED_TASKS
    config["task"] = task
    config["dmrgpy_out_dir"] = os.path.join(config["data_dir"], "dmrgpy_out")
    config["refine1_dir"] = os.path.join(config["data_dir"], "dmrgpy_out.1")
    config["sample_dir"] = os.path.join(config["data_dir"], "samples")
    basename_pattern = "dmrgpy_out_{seed:02d}_{lambda0i}_{lambda1i}.hdf5"
    if task == "dmrg":
        config["output_dir"] = config["dmrgpy_out_dir"]
    elif task == "refine1":
        config["output_dir"] = config["refine1_dir"]
        config["in_file_pattern"] = os.path.join(
            config["dmrgpy_out_dir"], basename_pattern)
    else:
        config["output_dir"] = config["sample_dir"]
        assert task in ("sample", "sample*")
        config["in_file_pattern"] = os.path.join(
            config["refine1_dir"], basename_pattern)
    config["output_file_pattern"] = os.path.join(
        config["output_dir"], basename_pattern)
    config["all_seeds"] = [3, 4]
    all_params = [
        (seed, lambda0i, lambda1i)
        for seed in config["all_seeds"]
        for lambda0i in range(LAMBDA0I_MAX + 1)
        for lambda1i in range(LAMBDA1I_MAX + 1)]
    # First, shuffle all_params in the same order for all tasks within array:
    order_seed = hash((os.environ.get("SLURM_ARRAY_TASK_ID", 0), args.num_jobs))
    # Ensure order_seed > 0:
    order_seed = 2 * abs(order_seed) + (order_seed >= 0)
    rng = np.random.default_rng(seed=order_seed)
    all_params = [all_params[i] for i in rng.permutation(len(all_params))]
    my_range_start = (args.job_id * len(all_params)) // args.num_jobs
    my_range_end = ((args.job_id + 1) * len(all_params)) // args.num_jobs
    my_params = all_params[my_range_start:my_range_end]
    # Shuffle the rest:
    other_params = all_params[:my_range_start] + all_params[my_range_end:]
    rng = np.random.default_rng(seed=args.num_jobs**2 + args.job_id)
    other_params = [other_params[i] for i in rng.permutation(len(other_params))]
    my_params = my_params + other_params
    config["my_params_list"] = my_params
    config["num_cpu_cores"] = get_num_cpu_cores(verbose=True)
    return config

def decode_params(task, params_tuple):
    seed, lambda0i, lambda1i = params_tuple
    res = {
        "seed": seed,
        "lambda0i": lambda0i,
        "lambda1i": lambda1i,
        "Jprime": 0 + 2.5 * lambda0i / 63,
        "delta": 0 + 3.5 * lambda1i / 63,
        "version": (seed - 1) % 2 + 1} # 1 or 2
    seed_gen = classifim_utils.DeterministicPrng(seed)
    if task in ["dmrg", "refine1"]:
        res["dmrg_seed"] = seed_gen.get_int64_seed((lambda0i, lambda1i))
    else:
        assert task in ("sample", "sample*")
        res["sample_seed"] = seed_gen.get_int64_seed(
            ("sample", lambda0i, lambda1i))
    return res

def run_elben_xxz_dmrg(config, version, dmrg_seed, Jprime, delta):
    jl = dmrgpy.juliarun.get_jlsession().jlsession
    run_fs = {
        1: jl.MatrixProductStates.run_elben_xxz_dmrg_v1,
        2: jl.MatrixProductStates.run_elben_xxz_dmrg_v2}
    res = run_fs[version](
        config["n_sites"], Jprime, delta, seed=dmrg_seed)
    return res

REFINE1_DMRG_CUTOFFS = {
    1: 1e-13,
    2: 1e-12}

REFINE1_NORM_CUTOFFS = {
    1: 1e-4,
    2: 1e-5}

def refine1_elben_xxz_dmrg(config, version, prev_res, dmrg_seed, Jprime, delta):
    jl = dmrgpy.juliarun.get_jlsession().jlsession
    new_res = jl.MatrixProductStates.refine1_elben_xxz_dmrg(
        prev_res,
        dmrg_cutoff=REFINE1_DMRG_CUTOFFS[version],
        norm_cutoff=REFINE1_NORM_CUTOFFS[version],
        seed=dmrg_seed)
    return new_res

def _overlap_with(jl_mps, my_res, other_filename):
    other_res = load_res(other_filename)
    my_states = my_res.states
    other_states = other_res.states
    return [[jl_mps.ITensors.inner(my_state, other_state)
            for other_state in other_states]
           for my_state in my_states]

def sample_and_overlap(config, prev_res, params, other_seed=None):
    task = config["task"]
    jl = dmrgpy.juliarun.get_jlsession().jlsession
    jl_mps = jl.MatrixProductStates
    psi = prev_res.states[0]
    res = {}
    if task != "sample*":
        rng = jl_mps.Random.MersenneTwister(params["sample_seed"])
        res["z_samples"] = jl_mps.sample_z(rng, psi, 1400)
        rng = jl_mps.Random.MersenneTwister(params["sample_seed"])
        res["pauli_samples"] = jl_mps.sample_pauli(rng, psi, 1400)
        if other_seed is not None:
            other_filename = config["in_file_pattern"].format(
                **(params | {"seed": other_seed}))
            if not os.path.exists(other_filename):
                print(f"Skipping other_seed={other_seed} because {other_filename} "
                      + "does not exist.", file=sys.stderr)
            else:
                res["seed_overlap.key"] = other_seed
                res["seed_overlap.value"] = jl.Array(np.array(_overlap_with(
                    jl_mps, prev_res, other_filename)))
        res["level_overlap"] = jl_mps.ITensors.inner(psi, prev_res.states[1])
    lo_keys = []
    lo_values = []
    for dlambda in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        other_lambda_is = [
                params[f"lambda{j}i"] + dlambda[j] for j in [0, 1]]
        other_filename = config["in_file_pattern"].format(
            **(params | {f"lambda{j}i": other_lambda_is[j] for j in [0, 1]}))
        if not os.path.exists(other_filename):
            print(f"Skipping dlambda={dlambda} because {other_filename} "
                  + "does not exist.", file=sys.stderr)
            continue
        lo_keys.append(other_lambda_is)
        lo_values.append(_overlap_with(jl_mps, prev_res, other_filename))
    res["lambda_overlap.keys"] = jl.Array(np.array(lo_keys))
    res["lambda_overlap.values"] = jl.Array(np.array(lo_values))
    return res

def run_task(config, params, filenames):
    task = config["task"]
    if task == "dmrg":
        res = run_elben_xxz_dmrg(
            config,
            version=params["version"],
            dmrg_seed=params["dmrg_seed"],
            **{key: params[key] for key in ["Jprime", "delta"]})
        return {"elben_xxz_dmrg_res": res}
    prev_res = load_res(filenames["in"])
    if task == "refine1":
        res = refine1_elben_xxz_dmrg(
            config,
            version=params["version"],
            prev_res=prev_res,
            dmrg_seed=params["dmrg_seed"],
            **{key: params[key] for key in ["Jprime", "delta"]})
        return dict(res)
    assert task in ("sample", "sample*")
    next_seed = params["seed"] + 1
    if next_seed not in config["all_seeds"]:
        next_seed = None
    res = sample_and_overlap(
        config,
        prev_res=prev_res,
        params=params,
        other_seed=next_seed)
    return dict(res)

@functools.cache
def lock_contents(num_jobs, job_id):
    # We want to include enough information to be able to
    # identify the process that created it, and to be able to verify
    # whether the lock was abandoned.
    slurm_vars = [
        f"{key}={os.environ[key]}"
        for key in [
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"]
        if key in os.environ]
    return (
        f"{num_jobs}.{job_id}\n"
        + ",".join(slurm_vars)
        + f",PID={os.getpid()},HOSTNAME={socket.gethostname()}\n")

def check_prereq_files(prohibited=[], required=[]):
    for file_name in prohibited:
        if os.path.exists(file_name):
            return False
    for file_name in required:
        if not os.path.exists(file_name):
            return False
    return True

def try_lock(lock_name, prohibited=[], required=[]):
    try:
        with open(lock_name, "x") as f:
            f.write(lock_contents(ARGS.num_jobs, ARGS.job_id))
        if not check_prereq_files(prohibited=prohibited, required=required):
            os.remove(lock_name)
            print(
                f"Race condition detected after creating {lock_name}.",
                file=sys.stderr, flush=True)
            return False
    except FileExistsError:
        print(
            f"Race condition detected while creating {lock_name}.",
            file=sys.stderr, flush=True)
        return False
    return True

def load_res(in_filename):
    jl = dmrgpy.juliarun.get_jlsession().jlsession
    jl_mps = jl.MatrixProductStates
    return jl_mps.read_elben_xxz_dmrg_res(in_filename, "elben_xxz_dmrg_res")

def initialize_filenames(config, params):
    required = []
    prohibited = []
    filenames = {}
    filenames["out"] = config["output_file_pattern"].format(**params)
    assert filenames["out"].endswith(".hdf5")
    filenames["tmp"] = filenames["out"] + ".tmp"
    filenames["lock"] = filenames["out"] + ".lock"
    if config["task"] in ["dmrg", "refine1"]:
        filenames["out_light"] = filenames["out"][:-5] + ".light.hdf5"
        filenames["tmp_light"] = filenames["out_light"] + ".tmp"
    for key in ["out", "tmp", "out_light", "tmp_light"]:
        if key in filenames:
            prohibited.append(filenames[key])
    if "in_file_pattern" in config:
        filenames["in"] = config["in_file_pattern"].format(**params)
        required.append(filenames["in"])
    return filenames, required, prohibited

def compute_time_guesstimate(config, params, filenames, data):
    task = config["task"]
    if task not in ("sample", "sample*"):
        return 0
    core_coeff = (1.0 + 10.0 / config["num_cpu_cores"]) / (1.0 + 10.0 / 12)
    task = config["task"]
    input_size = 0
    if "in" in filenames:
        try:
            input_size = os.path.getsize(os.path.realpath(filenames["in"]))
        except (FileNotFoundError, OSError):
            pass
    MAX_TIME = 800.0
    MAX_SIZE = 688e6
    res = core_coeff * (32.0 + MAX_TIME * input_size / MAX_SIZE)
    if task == "sample*":
        res /= 5
    data["time_estimate"] = res
    data["num_cpu_cores"] = config["num_cpu_cores"]
    data["input_size"] = input_size
    return res

def write_result(jl_mps, task, filenames, data):
    jl_mps.write_hdf5(filenames["tmp"], data)
    os.rename(filenames["tmp"], filenames["out"])
    print(f"Wrote {filenames['out']}.", file=sys.stderr, flush=True)
    if task in ["dmrg", "refine1"]:
        jl.empty_b(data["elben_xxz_dmrg_res"].states)
        jl_mps.write_hdf5(filenames["tmp_light"], data)
        os.rename(filenames["tmp_light"], filenames["out_light"])
        print(f"Wrote {filenames['out_light']}.", file=sys.stderr, flush=True)

STATUS_DONE = 0
STATUS_NOOP = 1
STATUS_SKIPPED = 2
def process_one_param_seed(config, params_tuple, remaining_time=None):
    task = config["task"]
    params = decode_params(task, params_tuple)
    filenames, required_filenames, prohibited_filenames = (
            initialize_filenames(config, params))
    task_name = (f"{task}("
        + ", ".join(
            f"{k}={params[k]}" for k in ["seed", "lambda0i", "lambda1i"])
        + ")")
    time_estimate = 0
    if not check_prereq_files(
            prohibited=[filenames["lock"]] + prohibited_filenames,
            required=required_filenames):
        return STATUS_NOOP
    data = {}
    if remaining_time is not None:
        time_estimate = compute_time_guesstimate(
                config, params, filenames, data)
        if time_estimate > remaining_time:
            print(
                f"Skipping {task_name}: "
                + f"{time_estimate} > {remaining_time}.",
                file=sys.stderr, flush=True)
            return STATUS_SKIPPED
    if not try_lock(
            filenames["lock"],
            prohibited=prohibited_filenames,
            required=required_filenames):
        return STATUS_NOOP
    try:
        time_start = time.time()
        datetime_start = datetime.datetime.fromtimestamp(time_start)
        print(
            f"{datetime_start:%Y-%m-%d %H:%M:%S}: {task_name}"
            + (f" ({time_estimate=:.1f}s)" if time_estimate else ""),
            file=sys.stderr, flush=True)
        data.update(run_task(config, params, filenames))
        sys.stdout.flush()
        jl = dmrgpy.juliarun.get_jlsession().jlsession
        jl_mps = jl.MatrixProductStates
        for key in ["seed", "lambda0i", "lambda1i", "version",
                "dmrg_seed", "sample_seed"]:
            if key in params:
                data[key] = params[key]
        data["task"] = task
        data["time"] = time.time() - time_start
        write_result(jl_mps, task, filenames, data)
        return STATUS_DONE
    finally:
        os.remove(filenames["lock"])

def create_dir(path, max_levels=2):
    if os.path.exists(path):
        return
    if max_levels <= 0:
        raise ValueError(f"Directory '{path}' does not exist.")
    try:
        os.mkdir(path)
    except FileNotFoundError:
        create_dir(os.path.dirname(path), max_levels=max_levels - 1)
        os.mkdir(path)

def main():
    # Turn off buffering for stdout and stderr:
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), 'wb', 0), write_through=True)
    sys.stderr = io.TextIOWrapper(
        open(sys.stderr.fileno(), 'wb', 0), write_through=True)

    print(f"Started at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}.",
          file=sys.stderr)
    config = init_config(ARGS)
    create_dir(config["output_dir"])
    remaining_time = None
    num_skipped = 0
    for params_tuple in config["my_params_list"]:
        status = process_one_param_seed(
                config, params_tuple, remaining_time=remaining_time)
        if status == STATUS_SKIPPED:
            num_skipped += 1
        if status != STATUS_DONE:
            continue
        cur_time = time.time()
        remaining_time = config["soft_deadline"] - cur_time
        if remaining_time < 0:
            print(
                f"Soft deadline reached: "
                + f"{cur_time} > {config['soft_deadline']}.",
                file=sys.stderr)
            break
    else:
        print(
            "Done" + (f" ({num_skipped} skipped)" if num_skipped else "."),
            file=sys.stderr)
    print(f"Finished at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}.",
          file=sys.stderr)

if __name__ == "__main__":
    main()
