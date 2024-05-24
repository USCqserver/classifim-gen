import numpy as np
import classifim.io
import classifim_gen.io

def convert_dataset_to_hf(seed, data, train_filename=None, test_filename=None):
    """
    Converts XXZ dataset from .npz to format compatible with HuggingFace.

    Args:
        seed (int): Seed for the random number generator.
        data: either
            - str: Path to the input .npz file containing the dataset
                with keys specified below.
            - dict: Dictionary with the following keys:
                - 'lambdas' (np.ndarray): Array of floats.
                - 'zs' (np.ndarray): Array of uint8s or int8s.
                - 'packed' (bool): Whether the samples are packed.
                    Only unpacked samples are supported.
                - [optional] scalar keys in [
                    'size_per_lambda', 'seed', 'sample_type'].
        train_filename (str): Path to the output train file.
        test_filename (str): Path to the output test file.
    """
    if isinstance(data, str):
        filename_in = data
        with np.load(filename_in) as npz:
            data = dict(npz)
    assert data["packed"] == False, "Only unpacked samples are supported."
    scalar_keys = [
        "size_per_lambda", "seed", "sample_type", "packed", "num_sites"]
    samples = data["zs"]
    num_samples, num_sites = samples.shape
    lambdas = data["lambdas"]
    assert lambdas.shape == (num_samples, 2)
    d_all = {
        "lambda0s": lambdas[:, 0],
        "lambda1s": lambdas[:, 1],
        "samples": classifim.io.samples1d_uint8_to_bytes(samples),
        "num_sites": num_sites}
    assert data.get("num_sites", num_sites) == num_sites
    for key in scalar_keys:
        if key in data:
            d_all[key] = data[key]
    prng = classifim.utils.DeterministicPrng(seed)
    d_train, d_test = classifim.io.split_train_test(
            d_all,
            test_size=0.1,
            seed=prng.get_int64_seed("split_test"),
            scalar_keys=scalar_keys)
    if train_filename is not None:
        classifim_gen.io.isnnn_dataset_save(d_train, train_filename)
    if test_filename is not None:
        classifim_gen.io.isnnn_dataset_save(d_test, test_filename)
    return d_train, d_test

