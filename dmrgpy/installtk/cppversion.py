# SPDX-License-Identifier: GPL-3.0-or-later
import re
import subprocess
import sys

def cppversion(gpp="g++"):
    """
    Returns the version of the installed g++ compiler as a list of integers.

    Args:
        gpp (str): The g++ executable to use. Default is 'g++'.

    Returns:
        A list of integers representing the version number,
        e.g., [12, 3, 1] for g++ 12.3.1.

    Raises:
        FileNotFoundError: If the g++ executable is not found.
        subprocess.CalledProcessError: If an error occurs while executing
            the g++ command.
        ValueError: If the version number cannot be parsed.
    """
    # Run the g++ command to get the version
    output = subprocess.check_output([gpp, "--version"], text=True)

    # Use regular expression to extract the version number
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", output)
    if match:
        # Convert the version parts to integers and return them
        return [int(part) for part in match.groups()]
    else:
        raise ValueError("Unable to parse g++ version number.")


def correct_version(verbose=False, **kwargs):
    try:
        version = cppversion(**kwargs)
    except Exception as err:
        if verbose:
            print(f"Error: {err}", file=sys.stderr)
        return False
    if version[0]>=6:
        return True
    else:
        if verbose:
            print(
                f"C++ compiler version {version} is too old.", file=sys.stderr)
        return False


if __name__=="__main__":
    print(cppversion())
    print(correct_version())
