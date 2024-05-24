import argparse
import datetime
import io
import json
import numpy as np
import os
import os.path
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


ARG_PRESETS = [
    ("thor_defaults",
        "Use default values for the Thor computer",
        {
            "sys_paths": [
                "~/.ipython/",
                "~/d/work/qc4ml/bs_chifc/classifim/src/"],
            "data_dir0": os.path.expanduser(
                "/media/victor/ssd_T7_Victor/d/qc4ml/bschifc_data/neurips2023"),
            "experiment_i": 0,
            "num_epochs": 2 # Testing locally.
        })]

def parse_arguments():
    parser = argparse.ArgumentParser(
            prog=f"python3 {os.path.basename(__file__)}",
            description=("Run xxz model training experiments."))
    for name, help_str, _ in ARG_PRESETS:
        parser.add_argument(
            f"--{name}", dest=name, default=False, action="store_true",
            help=help_str)
    parser.add_argument(
        "--sys_paths", type=str, default="",
        help="Comma-separated list of paths to add to sys.path.")
    parser.add_argument(
        "--data_dir0", type=str,
        help="Root directory for data files.")
    parser.add_argument(
        "--experiment_i", type=int,
        help="Index of the experiment.")
    parser.add_argument(
        "--num_epochs", type=int,
        help="Number of epochs.")
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

import classifim
import classifim_utils
from classifim.xxz1d import Pipeline

def init_config(args):
    data_dir = os.path.join(args.data_dir0, "xxz1d")
    assert os.path.isdir(data_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(
        f"Cuda is" + ("" if torch.cuda.is_available() else " NOT")
        + f" available; setting device = '{device}'.")
    models_dir = classifim_utils.maybe_create_subdir(data_dir, "models")
    fim_dir = classifim_utils.maybe_create_subdir(data_dir, "fim")
    model_name = "simple_conv1d"
    # experiments 0..9 are for z_6,...,z_15
    # experiments 10..19 are for pauli_6,...,pauli_15
    suffix1 = (
        ["z", "pauli"][args.experiment_i // 10] + "_"
        + str(args.experiment_i % 10 + 6))
    num_epochs = args.num_epochs or 580
    suffix2 = f"_e{num_epochs}"
    suffix = suffix1 + suffix2
    config = {
        "model_name": model_name,
        "dataset_filename": os.path.join(
            data_dir, "classifim_datasets", f"dataset_{suffix1}.npz"),
        "model_filename": os.path.join(
            models_dir, f"{model_name}_{suffix}.pth"),
        "log_filename": os.path.join(
            models_dir, f"{model_name}_{suffix}.log.json"),
        "fim_filename": os.path.join(fim_dir, f"{model_name}_{suffix}.npz"),
        # Note: suffix is a bit of a misnomer here. It is used to seed rng
        # including the rng for train/test split, hence should be the same
        # for all models (thus include only suffix1 and not suffix2):
        "suffix": suffix1,
        "scalar_keys": ["size_per_lambda", "seed", 'packed', 'sample_type'],
        "hold_out_test": False,
        "device": device,
        "num_epochs": num_epochs,
        "weight_decay": 5e-5
    }
    return config

def run_pipeline(config):
    t0 = time.time()
    cur_log = {"config": config}
    print(f"{datetime.datetime.now()}: "
          + f"{config['model_name']}_{config['suffix']}")
    pipeline = Pipeline(config=config)
    t1 = time.time()
    cur_log["train"] = pipeline.train()
    t2 = time.time()
    print(f"{datetime.datetime.now()}: Save")
    pipeline.save_model()
    pipeline.cleanup_after_training()
    pipeline.load_model()
    t3 = time.time()
    print(f"{datetime.datetime.now()}: Test")
    cur_log["test"] = pipeline.test(
        num_epochs=min(10, config["num_epochs"]))
    t4 = time.time()
    pipeline.eval_fim()
    pipeline.save_fim()
    t5 = time.time()
    dtrun = t5 - t0
    dttest = t4 - t3
    cur_log["timings"] = {
        "dtrun": dtrun,
        "dtrun_only": dtrun - dttest,
        "dttrain": t2 - t1,
        "dttest": dttest}
    print(
        "Timings: "
        + ", ".join(
            f"{key}: {value:.3f}s"
            for key, value in cur_log["timings"].items()))
    with open(config["log_filename"], "w") as f:
        json.dump(cur_log, f)
    print(f"{datetime.datetime.now()}: Done")

def main():
    # Turn off buffering for stdout and stderr:
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), 'wb', 0), write_through=True)
    sys.stderr = io.TextIOWrapper(
        open(sys.stderr.fileno(), 'wb', 0), write_through=True)

    config = init_config(ARGS)
    run_pipeline(config)

if __name__ == "__main__":
    main()
