"""Utilities for generating and processing Kitaev20 datasets."""

import classifim_gen.hubbard_hamiltonian
import numpy as np
import os

def generate_kitaev20_datasets(
        fermft_out, num_datasets=10, size_per_lambda=140,
        subset=None, payload=None):
    """
    Generate multiple Kitaev20 datasets for ClassiFIM method.

    Args:
        fermft_out: dict with keys:
            param.mus: the only lambda parameter (shape: (num_param_points,)).
            samples: pre-generated samples (shape: (num_param_points, num_samples)).
        num_datasets: Number of datasets to generate.
        size_per_lambda: Number of samples per row of lambdas (integer).
        subset: If provided, only use a subset of param_points (boolean array).
        payload: dict or a list of dicts of length num_datasets,
            which will be included in the returned datasets.

    Returns: List of datasets. Each dataset is a dict with the following keys:
        lambdas: (num_rows, 1) array of lambdas, where
            num_rows = num_param_points' * size_per_lambda,
            num_param_points' is num_param_points if subset is None,
            sum(subset) otherwise.
        packed_zs: (num_rows,) array of samples.
        dataset_i: Index of the dataset.
        num_datasets: Total number of datasets.
    """
    mus = fermft_out["param.mus"]
    samples = fermft_out["samples"]
    if subset is not None:
        mus = mus[subset]
        samples = samples[subset]
    num_param_points = mus.shape[0]
    num_samples = samples.shape[1]
    assert samples.shape == (num_param_points, num_samples)
    assert len(mus.shape) == 1
    assert size_per_lambda * num_datasets <= num_samples
    res = []
    for dataset_i in range(num_datasets):
        lambdas = np.repeat(mus[:, None], size_per_lambda, axis=0)
        sample_istart = dataset_i * size_per_lambda
        sample_iend = (dataset_i + 1) * size_per_lambda
        packed_zs = samples[:, sample_istart:sample_iend].reshape((-1,))
        cur_res = {
            "lambdas": lambdas,
            "packed_zs": packed_zs,
            "dataset_i": dataset_i,
            "num_datasets": num_datasets
        }
        if payload is not None:
            if isinstance(payload, dict):
                cur_res.update(payload)
            else:
                assert len(payload) == num_datasets
                cur_res.update(payload[dataset_i])
        res.append(cur_res)
    return res

def kitaev20_dataset_save(dataset, filename):
    """
    Save a dataset in Kitaev20 format:
    * one parameter (lambda0),
    * np.int32 samples.
    """
    return classifim_gen.hubbard_hamiltonian.hubbard12_dataset_save(
        dataset, filename, num_lambdas=1)

def convert_dataset_to_hf(
        seed, data, train_filename=None, test_filename=None):
    """
    Converts Kitaev20 dataset from .npz to format compatible with HuggingFace.
    """
    return classifim_gen.hubbard_hamiltonian.convert_dataset_to_hf(
        seed, data, train_filename=train_filename, test_filename=test_filename,
        num_lambdas=1, samples_column_name="packed_zs", samples_dtype=None,
        scalar_keys="all")
