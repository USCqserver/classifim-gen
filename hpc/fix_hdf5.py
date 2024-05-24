import os
import glob
import h5py
from tqdm import tqdm

def copy_hdf5_r(in_dir, out_dir, total=None):
    assert os.path.realpath(in_dir) != os.path.realpath(out_dir)
    # Iterate over all HDF5 files in the input directory
    for in_file in tqdm(glob.glob(os.path.join(in_dir, "*.hdf5")), total=total):
        # Extract the filename
        filename = os.path.basename(in_file)

        # Construct the corresponding output file path
        out_file = os.path.join(out_dir, filename)

        # Check if the output file exists
        if not os.path.exists(out_file):
            raise FileNotFoundError(f"No corresponding file found for {in_file} in {out_dir}")

        # Call the fix_hdf5 function
        copy_hdf5(in_file, out_file)
        with open(in_file + ".copied", "w") as f:
            pass

def copy_hdf5(in_file, out_file):
    """
    Copies all datasets from in_file to out_file, overwriting if necessary.

    Args:
        in_file: The input HDF5 file.
        out_file: The output HDF5 file.
            Datasets which do not match the keys of the input file will be
            preserved.
    """
    with h5py.File(in_file, 'r') as infile:
        with h5py.File(out_file, 'a') as outfile:
            for key in infile.keys():
                if key in outfile:
                    del outfile[key]
                infile.copy(key, outfile)


if __name__ == "__main__":
    copy_hdf5_r(
        in_dir=os.path.expanduser(
            "~/d/work/qc4ml/bschifc_data/neurips2023/xxz1d/samples/"),
        out_dir=os.path.expanduser(
            "~/d/work/qc4ml/bschifc_data/neurips2023/xxz1d/samples.1/"),
        total=2 * 64 * 64)
