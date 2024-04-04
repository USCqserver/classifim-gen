"""Utilities for generating and processing MNIST_CNN datasets."""

import numpy as np
import os
import scipy.special
from classifim.utils import load_tensor

def store_tensor_subset(orig_path, subset_ii, dest_path, tensor_name):
    """
    Select a subset of a tensor and store it in a new file.

    If the data is stored at dest_path already, it checks it is the same.

    Args:
        orig_path: Path to the original tensor.
        subset_ii: Indices of the subset.
        dest_path: Path to the destination file (.npz).
        tensor_name: Name of the tensor to be stored.
    """
    tensor = load_tensor(orig_path, tensor_name)
    tensor = tensor[subset_ii]
    assert dest_path.endswith(".npz")
    if not os.path.exists(dest_path):
        np.savez(dest_path, **{
            tensor_name: tensor,
            "subset_ii": subset_ii
        })
        return

    npz = dict(np.load(dest_path).items())
    # There are 2 valid cases:
    # 1. Both tensor_name and subset_ii are in npz, and they match.
    # 2. tensor_name is not in npz. If subset_ii is in npz, it must match.
    has_subset_ii = "subset_ii" in npz
    if has_subset_ii:
        # Check both shape and values:
        assert np.array_equal(npz["subset_ii"], subset_ii)
    else:
        npz["subset_ii"] = subset_ii
    if tensor_name in npz:
        assert has_subset_ii
        assert np.array_equal(npz[tensor_name], tensor)
    else:
        npz[tensor_name] = tensor
    np.savez(dest_path, **npz)

def generate_mnist_datasets(
        model_outputs, lambdas, seed=42, num_datasets=1, size_per_lambda=140,
        num_classes=10, idx=None, payload=None):
    """
    Generate multiple MNIST_CNN datasets for ClassiFIM method.

    Args:
        model_outputs: np.ndarray of shape (num_models, num_train_rows,
            num_classes) of outputs of MNIST_CNN models.
        lambdas: np.ndarray of lambda values.
        seed: Seed for random number generator.
        num_datasets: Number of datasets to generate.
        size_per_lambda: Number of samples per row of lambdas (integer).
        num_classes: Number of classes in the dataset (10 for MNIST).
        idx: Index conversion.
        payload: dict or a list of dicts of length num_datasets,
            which will be included in the returned datasets.

    Returns:
        List of datasets.
    """
    num_train_rows = model_outputs.shape[1]
    total_sample_size = size_per_lambda * num_datasets
    num_models = model_outputs.shape[0]
    assert lambdas.shape[0] == num_models, f"{lambdas.shape[0]} != {num_models}"
    rng = np.random.default_rng(seed)
    zs_ids = np.empty((num_models, total_sample_size), dtype=np.int32)
    for i in range(num_models):
        zs_ids[i] = rng.choice(num_train_rows, total_sample_size, replace=False)
    model_ii = np.arange(num_models)
    zs_probs = model_outputs[model_ii[:, None], zs_ids] # not probs yet
    zs_probs -= scipy.special.logsumexp(zs_probs, axis=2, keepdims=True)
    zs_probs = np.exp(zs_probs) # probs now
    # This is technically not necessary, but slightly reduces deviations
    # of sum from 1 from floating point errors:
    zs_probs /= np.sum(zs_probs, axis=2, keepdims=True)
    assert zs_probs.shape == (num_models, total_sample_size, num_classes)

    # Reshape everything to (num_datasets, num_models, size_per_lambda, *):
    zs_ids = zs_ids.reshape((num_models, num_datasets, size_per_lambda))
    zs_ids = zs_ids.transpose((1, 0, 2))
    zs_probs = zs_probs.reshape(
            (num_models, num_datasets, size_per_lambda, -1))
    zs_probs = zs_probs.transpose((1, 0, 2, 3))
    assert zs_probs.shape == (
        num_datasets, num_models, size_per_lambda, num_classes)
    lambdas = np.array(lambdas)
    assert len(lambdas.shape) == 2
    lambdas = np.repeat(lambdas[:, None, :], size_per_lambda, axis=1)
    res = []
    num_rows = num_models * size_per_lambda
    for i in range(num_datasets):
        cur_res = {
            "lambdas": lambdas.reshape((num_rows, 2)),
            "zs_ids": zs_ids[i].reshape((num_rows,)),
            "zs_probs": zs_probs[i].reshape((num_rows, num_classes)),
            "seed": seed,
            "num_datasets": num_datasets,
            "dataset_i": i,
            "size_per_lambda": size_per_lambda
        }
        if payload is not None:
            if isinstance(payload, dict):
                cur_res.update(payload)
            else:
                assert len(payload) == num_datasets
                cur_res.update(payload[i])
        res.append(cur_res)
    return res

class ParamConversions:
    """Class for converting between original parameters
    and their scaled forms for ClassiFIM method."""
    @staticmethod
    def learning_rates_to_lambdas(learning_rates):
        return np.log(learning_rates)

    @staticmethod
    def lambdas_to_learning_rates(lambdas):
        return np.exp(lambdas)

    @staticmethod
    def adam_beta1s_to_lambdas(beta1s):
        """Convert beta1s to lambdas.

        beta1 = 0 => lambda = 0
        beta1 = 1 => lambda = +inf
        """
        return -np.log(1 - beta1s)

    @staticmethod
    def lambdas_to_adam_beta1s(lambdas):
        """Convert lambdas to beta1s.

        lambda = 0 => beta1 = 0
        lambda = +inf => beta1 = 1
        """
        return 1 - np.exp(-lambdas)

    @staticmethod
    def construct_lambdas(learning_rates, beta1s):
        """Construct lambdas from learning rates and beta1s."""
        lambda0 = ParamConversions.learning_rates_to_lambdas(learning_rates)
        lambda1 = ParamConversions.adam_beta1s_to_lambdas(beta1s)
        return np.stack((lambda0, lambda1), axis=1)

