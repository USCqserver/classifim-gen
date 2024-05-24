import classifim.io
import glob
import json
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq

def isnnn_dataset_save(dataset, filename):
    """
    Save the dataset in the format of IsNNN400.

    Args:
        dataset (dict): Dictionary with the following keys:
            - 'lambda0s' (np.ndarray): Array of floats.
            - 'lambda1s' (np.ndarray): Array of floats.
            - 'samples' (np.ndarray): Array of bytes.
        filename (str): Path to the output file.
    """
    if filename.endswith(".parquet"):
        schema = pa.schema([
            pa.field("lambda0", pa.float32(), nullable=False),
            pa.field("lambda1", pa.float32(), nullable=False),
            pa.field(
                "sample",
                # We have a fixed-size column, so this would specify its size:
                # pa.binary(dataset["sample"].itemsize),
                # However, HuggingFace does not support fixed-size binary cols.
                pa.binary(),
                nullable=False),
        ])
        # Remove last 's' from keys:
        # Naming convention for features stored in npz is to use plural form
        # (e.g. lambda0s) to distinguish them from metadata (e.g. width).
        # On the other hand, storing
        # metadata in parquet columns is not supported (all parquet columns
        # have the same number of rows), so column name is feature name
        # (singular form).
        dataset = {
            k: dataset[k + 's'] for k in ['lambda0', 'lambda1', 'sample']}
        dataset['sample'] = classifim.io.bytes_to_pa(dataset['sample'])
        table = pa.Table.from_pydict(dataset, schema=schema)
        pq.write_table(table, filename)
    elif filename.endswith(".npz"):
        np.savez_compressed(filename, **dataset)
    else:
        raise ValueError(f"Unknown extension: {filename}")

def extract_meta_json(
        data, out=None, ignore_keys=None, extra_metadata=None):
    """
    Extract metadata from dataset.

    Args:
        data (dict): Metadata.
        out (str): Path to the output JSON file. If None, return the dict
            instead.
        ignore_keys (list): Keys to ignore. See the implementation for the
            default value.
        extra_metadata (dict): Extra metadata to save in the meta.json file.
    """
    if ignore_keys is None:
        ignore_keys = ["lambda0s", "lambda1s", "samples", "obss"]
    metadata = {}
    for key, value in data.items():
        if key in ignore_keys:
            continue
        if isinstance(value, np.ndarray):
            assert value.size == 1, (
                f"Only scalars are expected, got {key} "
                f"with shape {value.shape}")
            value = value.item()
        metadata[key] = value
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    if out is None:
        return metadata
    with open(out, "w") as f:
        json.dump(metadata, f)

def save_datasets_for_hf(
        convert_f, input_pattern: str, output_dir: str,
        extra_metadata: dict = None,
        overwrite: bool = False):
    """
    Go through datasets saved as npz files and convert them to HF format.

    Args:
        convert_f (Callable): Function to convert the dataset.
        input_pattern (str): Pattern for the input files. The pattern should
            contain the substring '{seed}' which will be replaced with the seed
            number.
        output_dir (str): Path to the output directory.
        extra_metadata (dict): Extra metadata to save in the meta.json file.
        overwrite (bool): Whether to overwrite existing files.
    """
    base_path, seed_pattern = input_pattern.split('{seed}')
    glob_pattern = f"{base_path}*{seed_pattern}"
    input_files = glob.glob(glob_pattern)
    input_files = [
        (int(input_file[len(base_path): -len(seed_pattern)]), input_file)
        for input_file in input_files]

    data_seeds = []
    res = None
    for seed, input_file in input_files:
        data_seeds.append(seed)
        output_subdir = classifim.utils.maybe_create_subdir(output_dir, f"seed_{seed:02d}")
        train_filename = os.path.join(output_subdir, "d_train.parquet")
        if not overwrite and os.path.exists(train_filename):
            train_filename = None
        test_filename = os.path.join(output_subdir, "d_test.parquet")
        if not overwrite and os.path.exists(test_filename):
            test_filename = None

        if train_filename is None and test_filename is None:
            print(f"Skipping seed {seed}: output files already exist.")
            continue

        res = convert_f(
            seed,
            input_file,
            train_filename=train_filename,
            test_filename=test_filename)

    if res is None:
        if len(input_files) == 0:
            raise ValueError(f"No input files matched '{glob_pattern}'.")
        seed, input_file = input_files[0]
        res = convert_f(seed, input_file)
        assert res is not None
    extract_meta_json(
        res[0],
        out=os.path.join(output_dir, "meta.json"),
        extra_metadata=extra_metadata)
    return data_seeds

def trim_multiline(s):
    """
    Trim a multiline string.

    This removes:
    - leading whitespace-only lines,
    - trailing whitespace,
    - indentation common to all non-empty lines.
    """
    # Step 1: remove trailing whitespace and split into lines:
    lines = s.rstrip().splitlines()
    min_indent = float('inf')

    # Step 2: Remove leading whitespace
    num_empty = len(lines)
    for i, line in enumerate(lines):
        if line.strip():
            num_empty = i
            break

    # Step 3: Find the common indentation
    for line in lines[num_empty:]:
        stripped_line = line.lstrip()
        if stripped_line:
            indent = len(line) - len(stripped_line)
            min_indent = min(min_indent, indent)

    # Step 4: Join and return
    return '\n'.join(line[min_indent:] for line in lines[num_empty:])

def gen_config_yml(sm_name, seeds, fim_seeds=None):
    """
    Generate HuggingFace configs yml (part of the dataset card).

    Args:
        sm_name (str): Name of the statistical manifold.
        seeds (list): List of seeds for datasets.
        fim_seeds (list): List of seeds with gt_fim. Special values:
            - None: no seed dependence of gt_fim (default).
            - "same": same as `seeds`.
            - []: no gt_fim.
    """
    res = []
    if fim_seeds is None:
        res.append(f"""
            - config_name: {sm_name}.gt_fim
              data_files:
              - split: test
                path: {sm_name}/gt_fim.parquet
            """)
    else:
        if fim_seeds == "same":
            fim_seeds = seeds
        for seed in fim_seeds:
            res.append(f"""
                - config_name: {sm_name}.seed{seed:02d}.gt_fim
                  data_files:
                  - split: test
                    path: {sm_name}/seed_{seed:02d}/gt_fim.parquet
                """)
    for seed in seeds:
        res.append(f"""
            - config_name: {sm_name}.seed{seed:02d}
              data_files:
              - split: train
                path: {sm_name}/seed_{seed:02d}/d_train.parquet
              - split: test
                path: {sm_name}/seed_{seed:02d}/d_test.parquet
            """)
    return '\n'.join(trim_multiline(x) for x in res)

