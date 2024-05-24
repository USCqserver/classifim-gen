#!/bin/python3

import argparse
import datetime
import numpy as np
import os.path
import sys
import time
import torch
from sklearn.preprocessing import MinMaxScaler

ARG_PRESETS = [
    ("thor_defaults",
        "Use default values for the Thor computer",
        {
            "sys_paths": [
                "~/.ipython/",
                "~/d/work/qc4ml/bs_chifc/bitstring-chifc/supplementary_materials/code/"],
            "data_dir0": os.path.expanduser(
                f"/run/media/victor/ssd_T7_Victor/d/qc4ml/bschifc_data/neurips2023")
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
    parser.add_argument("--suffix", type=str, default="",
            help=("Suffix for input and output file names."))
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

import classifim
import classifim.input
import classifim_utils
import classifim.mnist
from classifim.mnist import MnistZsTransform, load_mnist_data

def init_config(args):
    config_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Cuda is", end="")
    if not torch.cuda.is_available():
        print(" NOT", end="")
    print(f" available; setting config_device = '{config_device}'.")

    data_dir0 = args.data_dir0
    assert os.path.isdir(data_dir0), (
        f"Data directory '{data_dir0}' does not exist.")


    dataset_name = "mnist_cnn"
    data_dir = os.path.join(data_dir0, dataset_name.lower())
    datasets_dir = os.path.join(data_dir, "classifim_datasets")
    models_dir = classifim_utils.maybe_create_subdir(data_dir, "models")
    fim_dir = classifim_utils.maybe_create_subdir(data_dir, "fim")

    if ARGS.suffix:
        suffix = ARGS.suffix
    else:
        dataset_i = ARGS.job_id
        suffix = f"42a_{dataset_i}"
    hpc_out_suffix = f"hpc0001_{suffix}"
    config = {
        "suffix": suffix,
        "dataset_filename": os.path.join(datasets_dir, f"dataset_{suffix}.npz"),
        "model_filename": os.path.join(
            models_dir, f"9_reshuffle_{hpc_out_suffix}.pth"),
        "log_filename": os.path.join(
            models_dir, f"9_reshuffle_{hpc_out_suffix}.log.npz"),
        "fim_filename": os.path.join(
            fim_dir, f"9_reshuffle_{hpc_out_suffix}.npz"),
        "test_dump_filename": os.path.join(
            datasets_dir, f"test_dump_{hpc_out_suffix}.npz"),
        "num_epochs": 319,
        "batch_size": 2**13,
        "weight_decay": 1e-4,
        # hold_out_test=True means test set is not used. Instead,
        #   a separate validation set is split out from the training set for
        #   testing and validation.
        # hold_out_test=False means the test results are included in the output.
        "hold_out_test": False,
        "scalar_keys": set([
            "seed", "num_datasets", "dataset_i", "size_per_lambda",
            "model_out.seed", "mnist_labels_filename",
            "mnist_inputs_filename", "mnist_inputs", "mnist_labels"]),
        "device": config_device}

    assert os.path.isfile(config["dataset_filename"]), (
        f'File {config["dataset_filename"]} not found!')

    return config

def main():
    print(f"{datetime.datetime.now()}: Starting main()...")
    config = init_config(ARGS)
    npz_dataset, mnist_inputs, mnist_labels = load_mnist_data(
            config, verbose=True)
    dataset = dict(npz_dataset.items())
    dataset["mnist_inputs"] = mnist_inputs
    dataset["mnist_labels"] = mnist_labels
    pipeline = classifim.mnist.Pipeline(dataset=dataset, config=config)
    print(f"{datetime.datetime.now()}: Train:", flush=True)
    train_log = pipeline.train()
    print(f"{datetime.datetime.now()}: Saving model:", flush=True)
    pipeline.save_model()
    print(f"{datetime.datetime.now()}: Test:", flush=True)
    sys.stdout.flush()
    test_log = pipeline.test()
    print(f"{datetime.datetime.now()}: Computing FIM:", flush=True)
    pipeline.eval_fim()
    pipeline.save_fim()
    print(f"{datetime.datetime.now()}: Saving logs:", flush=True)
    log = {"train": train_log, "test": test_log}
    log = classifim.mnist.flatten_dict_for_npz(log)
    np.savez(config["log_filename"], **log)
    print(f"{datetime.datetime.now()}: Done.")

if __name__ == "__main__":
    main()
