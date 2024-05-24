# SPDX-License-Identifier: GPL-3.0-or-later
import os
from . import taskdmrg

dmrgpath = os.path.dirname(os.path.realpath(__file__)) # path for the program
mpscpp = dmrgpath+"/mpscpp2/mpscpp.x" # C++ executable

def run(self, automatic=False):
    """
    Run the DMRG calculation
    """
    if get_mode(self)=="ED":
        return # do nothing
    self.execute(lambda : taskdmrg.write_tasks(self)) # write tasks
    if self.itensor_version in [2, "2", "v2", "C++", "cpp", "c", "C"]:
        if os.path.isfile(mpscpp):
            from . import cpprun
            cpprun.run(self) # run the C++ version
            return
    elif self.itensor_version in ["julia", "Julia", "jl"]:
        from . import juliarun
        juliarun.run(self)
        return
    else:
        raise ValueError(
            f"Unrecognized itensor_version '{self.itensor_version}'")

def get_mode(self, mode="DMRG"):
    """Return the mode of the calculation"""
    # if there is no C++, then use ED
    if not os.path.isfile(mpscpp):
        print("C++ not compiled, using default ED routines")
        return "ED" # use exact diagonalization

    # if there is an enforced mode, then use that one
    if self.mode is not None:
        return self.mode # use the enforced mode
    else:
        if mode in ["ED", "DMRG"]:
            return mode # use default
        else:
            raise ValueError("Unrecognized mode " + mode)


