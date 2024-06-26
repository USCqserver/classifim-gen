# SPDX-License-Identifier: GPL-3.0-or-later
# compute vacuum expectation values using multioperators

from . import multioperator

import numpy as np

def power_vev(self,wf=None,n=4,X=None,**kwargs):
    """Compute the moments of an operator"""
    if wf is None: wf = self.get_gs(**kwargs) # ground state
    wfs = [wf.copy()] # wavefunctions
    wfi = wf.copy()
    for i in range(n): # get X**n |WF>
        wfi = X*wfi
        wfs.append(wfi.copy()) # apply the operator
    out = [0.0j for i in range(2*(n//2))] # output
    for i in range(n//2): # loop over powers wanted
        out[2*i] = wfs[i].dot(wfs[i])
        out[2*i+1] = wfs[i].dot(wfs[i+1])
    return np.array(out)



def multi_vev(self,MO,wf=None,npow=1,**kwargs):
    """
    Compute a VEV using multioperators
    """
    MO = multioperator.obj2MO(MO,name="vev_multioperator")
    if MO.name!="vev_multioperator": raise
    if npow==0: return 1.0
    if wf is None: wf = self.get_gs() # get the ground state
    wf.write(name="wf_vev.mps") # write wavefunction
    taskd = MO.get_dict() # get the dictionary
    self.task["vev"] = "true" # do a VEV
    self.task["wf_vev"] = "wf_vev.mps" # WF to use VEV
    self.task["pow_vev"] = int(npow) # power
    self.write_task() # write the tasks in a file
    self.write_hamiltonian() # write the Hamiltonian to a file
    self.execute(lambda: MO.write()) # write multioperator
    self.run() # perform the calculation
    m = self.execute(lambda: np.genfromtxt("VEV.OUT"))
    return m[0]+1j*m[1] # return result


def vev(*args,**kwargs):
    return multi_vev(*args,**kwargs)


def excited_vev(*args,**kwargs):
    return multi_vev(*args,excited=True,**kwargs)












