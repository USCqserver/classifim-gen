import classifim.utils
import classifim_gen.io
import concurrent.futures
import ctypes
import functools
import itertools
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

from tqdm import tqdm

from numpy.ctypeslib import ndpointer

ISING_2D_BETA_CRITICAL = np.log(1 + np.sqrt(2)) / 2

class IsingMCMC2DBase:
    def __init__(self):
        self._lib = None
        self._lib = self._get_lib()

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_lib():
        extension = {
            "win32": ".dll",
            "darwin": ".dylib"
        }.get(sys.platform, ".so")
        lib_path = os.path.join(
            os.path.dirname(__file__),
            'lib',
            'libclassifim_gen' + extension)
        lib = ctypes.CDLL(lib_path)

        # Define functions here.
        # Base:
        lib.delete_ising_mcmc2D_base.argtypes = [ctypes.c_void_p]
        lib.delete_ising_mcmc2D_base.restype = None

        lib.ising_mcmc2D_base_get_state.argtypes = [
            ctypes.c_void_p,
            ndpointer(ctypes.c_uint64, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
        ]
        lib.ising_mcmc2D_base_get_state.restype = None

        lib.ising_mcmc2D_base_set_state.argtypes = [
            ctypes.c_void_p,
            ndpointer(ctypes.c_uint64, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
        ]
        lib.ising_mcmc2D_base_set_state.restype = None

        lib.ising_mcmc2D_base_produce_shifted_state.argtypes = [
            ctypes.c_void_p,
            ndpointer(ctypes.c_uint64, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
        ]
        lib.ising_mcmc2D_base_produce_shifted_state.restype = None

        lib.ising_mcmc2D_base_reset.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_reset.restype = None

        lib.ising_mcmc2D_base_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.ising_mcmc2D_base_step.restype = None

        lib.ising_mcmc2D_base_get_width.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_width.restype = ctypes.c_int

        lib.ising_mcmc2D_base_get_height.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_height.restype = ctypes.c_int

        lib.ising_mcmc2D_base_get_magnetization.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_magnetization.restype = ctypes.c_double

        lib.ising_mcmc2D_base_get_total_nn.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_total_nn.restype = ctypes.c_int

        lib.ising_mcmc2D_base_get_total_nnn.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_total_nnn.restype = ctypes.c_int

        lib.ising_mcmc2D_base_get_energy0.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_energy0.restype = ctypes.c_int

        lib.ising_mcmc2D_base_step_parallel_tempering.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]
        lib.ising_mcmc2D_base_step_parallel_tempering.restype = ctypes.c_int

        lib.ising_mcmc2D_base_get_beta_energy.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_beta_energy.restype = ctypes.c_double

        # MCMC 2D:
        lib.create_ising_mcmc2D.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double
        ]
        lib.create_ising_mcmc2D.restype = ctypes.c_void_p

        lib.ising_mcmc2D_adjust_parameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double]
        lib.ising_mcmc2D_adjust_parameters.restype = None

        lib.ising_mcmc2D_step_flip.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_step_flip.restype = None

        lib.ising_mcmc2D_get_beta.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_get_beta.restype = ctypes.c_double

        lib.ising_mcmc2D_get_h.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_get_h.restype = ctypes.c_double

        lib.ising_mcmc2D_step_combined_ef.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
            ctypes.c_bool]
        lib.ising_mcmc2D_step_combined_ef.restype = None

        # NNN MCMC:
        lib.create_ising_nnn_mcmc.argtypes = [
            ctypes.c_uint64,  # std::uint64_t seed
            ctypes.c_int,     # int width
            ctypes.c_int,     # int height
            ctypes.c_double,  # double beta
            ctypes.c_double,  # double jh
            ctypes.c_double,  # double jv
            ctypes.c_double,  # double jp
            ctypes.c_double,  # double jm
            ctypes.c_double   # double h
        ]
        lib.create_ising_nnn_mcmc.restype = ctypes.c_void_p

        lib.ising_nnn_mcmc_adjust_parameters.argtypes = [
            ctypes.c_void_p,  # classifim_gen::IsingNNNMCMC *mcmc
            ctypes.c_double,  # double beta
            ctypes.c_double,  # double jh
            ctypes.c_double,  # double jv
            ctypes.c_double,  # double jp
            ctypes.c_double,  # double jm
            ctypes.c_double   # double h
        ]
        lib.ising_nnn_mcmc_adjust_parameters.restype = None

        lib.ising_nnn_mcmc_step_flip.argtypes = [
            ctypes.c_void_p,  # classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_step_flip.restype = None

        lib.ising_nnn_mcmc_get_beta.argtypes = [
            ctypes.c_void_p,  # const classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_get_beta.restype = ctypes.c_double

        lib.ising_nnn_mcmc_get_jh.argtypes = [
            ctypes.c_void_p,  # const classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_get_jh.restype = ctypes.c_double

        lib.ising_nnn_mcmc_get_jv.argtypes = [
            ctypes.c_void_p,  # const classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_get_jv.restype = ctypes.c_double

        lib.ising_nnn_mcmc_get_jp.argtypes = [
            ctypes.c_void_p,  # const classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_get_jp.restype = ctypes.c_double

        lib.ising_nnn_mcmc_get_jm.argtypes = [
            ctypes.c_void_p,  # const classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_get_jm.restype = ctypes.c_double

        lib.ising_nnn_mcmc_get_h.argtypes = [
            ctypes.c_void_p,  # const classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_get_h.restype = ctypes.c_double


        lib.ising_nnn_mcmc_step_combined_2of.argtypes = [
            ctypes.c_void_p,  # classifim_gen::IsingNNNMCMC *mcmc
            ctypes.c_int,     # int n_steps
            ctypes.c_int,     # int n_obs_samples
            # std::int32_t *observables:
            ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
            ctypes.c_bool     # bool flip
        ]
        lib.ising_nnn_mcmc_step_combined_2of.restype = None

        lib.ising_nnn_mcmc_step_spins.argtypes = [ctypes.c_void_p]
        lib.ising_nnn_mcmc_step_spins.restype = None

        lib.ising_nnn_mcmc_step_lines.argtypes = [ctypes.c_void_p]
        lib.ising_nnn_mcmc_step_lines.restype = None

        lib.ising_nnn_mcmc_step_lines_horizontal.argtypes = [ctypes.c_void_p]
        lib.ising_nnn_mcmc_step_lines_horizontal.restype = None

        lib.ising_nnn_mcmc_step_lines_vertical.argtypes = [ctypes.c_void_p]
        lib.ising_nnn_mcmc_step_lines_vertical.restype = None

        return lib

    def __del__(self):
        if hasattr(self, '_mcmc') and self._mcmc is not None:
            self._lib.delete_ising_mcmc2D_base(self._mcmc)

    def get_state(self, out=None, shifted=False):
        height = self.get_height()

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError("The 'out' argument must be a numpy.ndarray.")
            if out.ndim != 1:
                raise ValueError("The 'out' array must be 1-dimensional.")
            if out.shape[0] != height:
                raise ValueError(
                    "The 'out' array must have exactly height elements.")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("The 'out' array must be C-contiguous.")
            if out.dtype != np.uint64:
                raise TypeError("The dtype of 'out' must be uint64.")
        else:
            out = np.empty(height, dtype=np.uint64)

        if shifted:
            f = self._lib.ising_mcmc2D_base_produce_shifted_state
        else:
            f = self._lib.ising_mcmc2D_base_get_state
        f(self._mcmc, out, ctypes.c_size_t(height))

        return out

    def _get_state(self, out, shifted=False):
        """Same as `get_state` but does not check the arguments.

        Also, the `out` argument is not returned.
        """
        if shifted:
            f = self._lib.ising_mcmc2D_base_produce_shifted_state
        else:
            f = self._lib.ising_mcmc2D_base_get_state
        f(self._mcmc, out, ctypes.c_size_t(out.shape[0]))

    def set_state(self, state):
        """
        Set the state of the 2D Ising model.

        Args:
            state: The state vector of the 2D Ising model.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError("The 'state' argument must be a numpy.ndarray.")
        if state.ndim != 1:
            raise ValueError("The 'state' array must be 1-dimensional.")
        if state.shape[0] != self.get_height():
            raise ValueError(
                "The 'state' array must have exactly height elements.")
        if not state.flags['C_CONTIGUOUS']:
            raise ValueError("The 'state' array must be C-contiguous.")
        if state.dtype != np.uint64:
            raise TypeError("The dtype of 'state' must be uint64.")

        self._lib.ising_mcmc2D_base_set_state(
            self._mcmc, state, ctypes.c_size_t(state.shape[0]))

    def reset(self):
        self._lib.ising_mcmc2D_base_reset(self._mcmc)

    def step(self, n_steps=1):
        self._lib.ising_mcmc2D_base_step(self._mcmc, ctypes.c_int(n_steps))

    def get_width(self):
        return self._lib.ising_mcmc2D_base_get_width(self._mcmc)

    def get_height(self):
        return self._lib.ising_mcmc2D_base_get_height(self._mcmc)

    def get_area(self):
        return self.get_width() * self.get_height()

    def get_magnetization(self):
        return self._lib.ising_mcmc2D_base_get_magnetization(self._mcmc)

    def get_total_nn(self):
        return self._lib.ising_mcmc2D_base_get_total_nn(self._mcmc)

    def get_total_nnn(self):
        return self._lib.ising_mcmc2D_base_get_total_nnn(self._mcmc)

    def get_energy0(self):
        """
        Returns the energy of the current state, assuming h=0.
        """
        return self._lib.ising_mcmc2D_base_get_energy0(self._mcmc)

    def step_parallel_tempering(self, other):
        return self._lib.ising_mcmc2D_base_step_parallel_tempering(
            self._mcmc, other._mcmc)

    def get_beta_energy(self):
        return self._lib.ising_mcmc2D_base_get_beta_energy(self._mcmc)

    @staticmethod
    def unpack_state_to_matrix(state, width):
        """
        Unpack the state vector to a 2D matrix.

        Args:
            state: The state vector of the 2D Ising model.
            width: The width of the 2D lattice.
        """
        jj = np.arange(width, dtype=np.uint64)
        return 1 - 2 * ((state[:, None] >> jj[None, :]) & 1).astype(np.int64)

    def get_state_matrix(self):
        """
        Returns the current state as a 2D matrix.
        """
        return self.unpack_state_to_matrix(self.get_state(), self.get_width())

class IsingMCMC2D(IsingMCMC2DBase):
    BETA_CRITICAL = ISING_2D_BETA_CRITICAL
    def __init__(self, seed=1, width=20, height=20,
                 beta=ISING_2D_BETA_CRITICAL, h=0.0):
        """
        Initialize the 2D Ising model MCMC sampler.

        Implemented in C++.

        Args:
            seed: The seed for random number generation, used in the MCMC steps.
            width: The width of the 2D lattice, between 2 and 62 (inclusive).
            height: The height of the 2D lattice, a positive integer >= 2.
            beta: The inverse temperature parameter of the Ising model.
            h: The external magnetic field.
        """
        super().__init__()

        _mcmc = self._lib.create_ising_mcmc2D(
            ctypes.c_uint64(seed),
            ctypes.c_int(width),
            ctypes.c_int(height),
            ctypes.c_double(beta),
            ctypes.c_double(h)
        )
        self._mcmc = ctypes.c_void_p(_mcmc)

    def adjust_parameters(self, beta=ISING_2D_BETA_CRITICAL, h=0.0):
        self._lib.ising_mcmc2D_adjust_parameters(
                self._mcmc, ctypes.c_double(beta), ctypes.c_double(h))

    def step_flip(self):
        self._lib.ising_mcmc2D_step_flip(self._mcmc)

    def get_beta(self):
        return self._lib.ising_mcmc2D_get_beta(self._mcmc)

    def get_h(self):
        return self._lib.ising_mcmc2D_get_h(self._mcmc)

    def get_energy(self, h):
        """
        Returns the energy of the current state, assuming h=h.
        """
        grid_size = self.get_width() * self.get_height()
        return (
            1.0 * self.get_energy0() - h * self.get_magnetization() * grid_size)

    def step_combined_ef(self, n_steps, energies_out, flip):
        """
        Perform n_steps of MCMC steps, and store the energies in energies_out.

        Args:
            n_steps: The number of MCMC steps to perform.
            energies_out: The array to store the energies.
            flip: If True, perform a flip at the end of the MCMC steps.
        """
        if not isinstance(energies_out, np.ndarray):
            raise TypeError("The 'energies_out' argument must be a numpy.ndarray.")
        if energies_out.ndim != 1:
            raise ValueError("The 'energies_out' array must be 1-dimensional.")
        if energies_out.shape[0] > n_steps:
            raise ValueError(
                "We can sample at most 1 energy per MCMC step.")
        if not energies_out.flags['C_CONTIGUOUS']:
            raise ValueError("The 'energies_out' array must be C-contiguous.")
        if energies_out.dtype != np.int32:
            raise TypeError("The dtype of 'energies_out' must be int32.")

        self._lib.ising_mcmc2D_step_combined_ef(
            self._mcmc, ctypes.c_int(n_steps),
            ctypes.c_int(energies_out.shape[0]),
            energies_out, ctypes.c_bool(flip))

    @classmethod
    def difficulty_factor(cls, t):
        """
        Returns heuristic difficulty factor to scale the number of MCMC steps.

        Args:
            t: The temperature (or np.ndarray of temperatures).
        """
        delta_t = t - 1.0 / cls.BETA_CRITICAL
        return 1.0 + 1.0 / (0.1 + delta_t**2)

    def cur_difficulty_factor(self):
        return self.difficulty_factor(1.0 / self.get_beta())

class IsingNNNMCMC(IsingMCMC2DBase):
    def __init__(self, seed=1, width=20, height=20,
                 beta=ISING_2D_BETA_CRITICAL, jh=1.0, jv=1.0, jp=0.0, jm=0.0, h=0.0):
        """
        Initialize the MCMC sampler for 2D Ising model with NNN interactions.

        Implemented in C++.

        Args:
            seed: The seed for random number generation, used in the MCMC steps.
            width: The width of the 2D lattice, between 2 and 62 (inclusive).
            height: The height of the 2D lattice, a positive integer >= 2.
            beta: The inverse temperature parameter of the Ising model.
            jh: The coupling strength for the horizontal interactions.
            jv: The coupling strength for the vertical interactions.
            jp: The coupling strength for the positive diagonal interactions.
            jm: The coupling strength for the negative diagonal interactions.
            h: The external magnetic field.
        """
        super().__init__()

        _mcmc = self._lib.create_ising_nnn_mcmc(
            ctypes.c_uint64(seed),
            ctypes.c_int(width),
            ctypes.c_int(height),
            ctypes.c_double(beta),
            ctypes.c_double(jh),
            ctypes.c_double(jv),
            ctypes.c_double(jp),
            ctypes.c_double(jm),
            ctypes.c_double(h)
        )
        self._mcmc = ctypes.c_void_p(_mcmc)

    def adjust_parameters(self, beta, jh, jv, jp, jm, h):
        self._lib.ising_nnn_mcmc_adjust_parameters(
            self._mcmc,
            ctypes.c_double(beta),
            ctypes.c_double(jh),
            ctypes.c_double(jv),
            ctypes.c_double(jp),
            ctypes.c_double(jm),
            ctypes.c_double(h))

    def adjust_some_parameters(self, **kwargs):
        self.adjust_parameters(
            beta=kwargs.get('beta', self.get_beta()),
            jh=kwargs.get('jh', self.get_jh()),
            jv=kwargs.get('jv', self.get_jv()),
            jp=kwargs.get('jp', self.get_jp()),
            jm=kwargs.get('jm', self.get_jm()),
            h=kwargs.get('h', self.get_h()))

    def step_flip(self):
        self._lib.ising_nnn_mcmc_step_flip(self._mcmc)


    def get_beta(self):
        return self._lib.ising_nnn_mcmc_get_beta(self._mcmc)

    def get_jh(self):
        return self._lib.ising_nnn_mcmc_get_jh(self._mcmc)

    def get_jv(self):
        return self._lib.ising_nnn_mcmc_get_jv(self._mcmc)

    def get_jp(self):
        return self._lib.ising_nnn_mcmc_get_jp(self._mcmc)

    def get_jm(self):
        return self._lib.ising_nnn_mcmc_get_jm(self._mcmc)

    def get_h(self):
        return self._lib.ising_nnn_mcmc_get_h(self._mcmc)

    def step_combined_2of(self, n_steps, observables_out=None, flip=False):
        """
        Perform n_steps of MCMC steps storing the 2 obs in observables_out.

        Args:
            n_steps: The number of MCMC steps to perform.
            observables_out: The array to store the observables.
            flip: If True, perform a flip at the end of the MCMC steps.
        """
        if observables_out is None:
            observables_out = np.empty(dtype=np.int32, shape=(n_steps, 2))
        if not isinstance(observables_out, np.ndarray):
            raise TypeError(
                "The 'observables_out' argument must be a numpy.ndarray.")
        if observables_out.ndim != 2:
            raise ValueError("The 'observables_out' must be 2-dimensional.")
        n_obs_samples = observables_out.shape[0]
        if n_obs_samples > n_steps:
            raise ValueError(
                "We can sample at most 1 energy per MCMC step.")
        if observables_out.shape[1] != 2:
            raise ValueError(
                "The 'observables_out' array must have 2 columns "
                "(one for each observable).")
        if observables_out.dtype != np.int32:
            raise TypeError("The dtype of 'observables_out' must be int32.")
        if not observables_out.flags['C_CONTIGUOUS']:
            raise ValueError("The 'observables_out' array must be C-contiguous.")

        self._lib.ising_nnn_mcmc_step_combined_2of(
            self._mcmc, ctypes.c_int(n_steps),
            ctypes.c_int(n_obs_samples),
            observables_out, ctypes.c_bool(flip))
        return observables_out

    def step_spins(self):
        self._lib.ising_nnn_mcmc_step_spins(self._mcmc)

    def step_lines(self):
        self._lib.ising_nnn_mcmc_step_lines(self._mcmc)

    def step_lines_horizontal(self):
        self._lib.ising_nnn_mcmc_step_lines_horizontal(self._mcmc)

    def step_lines_vertical(self):
        self._lib.ising_nnn_mcmc_step_lines_vertical(self._mcmc)

class PTArray:
    def __init__(self, mcmcs):
        """
        Initialize the PTArray with a list of IsingMCMC2DBase instances.

        Args:
            mcmcs (list): List of IsingMCMC2DBase instances.
        """
        assert len(mcmcs) >= 1
        width = mcmcs[0].get_width()
        height = mcmcs[0].get_height()
        for i in range(1, len(mcmcs)):
            assert mcmcs[i].get_width() == width
            assert mcmcs[i].get_height() == height
        self.mcmcs = mcmcs

    def __len__(self):
        return len(self.mcmcs)

    def __getitem__(self, i):
        return self.mcmcs[i]

    def step_all(self, n_steps=1, imax=None):
        """
        Perform the specified number of steps on all MCMC instances in parallel.

        Args:
            n_steps (int): Number of steps to perform.
            imax (int): If specified, only perform steps on mcmcs[:imax].
        """
        if imax is None:
            imax = len(self.mcmcs)
        for mcmc in self.mcmcs[:imax]:
            mcmc.step(n_steps)

    def step_pt(self, imin=0, imax=None, di=1):
        """
        Perform a parallel tempering step between pairs (
            self.mcmcs[imin + j], self.mcmcs[imin + di + j])
        for j in range(0, imax - imin - di).

        Args:
            imin (int): The starting index of the first element in the pair.
            imax (int): The maximum index to perform parallel tempering.
            di (int): The index difference between elements in the pair.
        """
        assert di > 0
        if imax is None:
            imax = len(self.mcmcs)
        assert imin + di <= imax <= len(self.mcmcs)
        for j in range(imin, imax - di):
            self.mcmcs[j].step_parallel_tempering(self.mcmcs[j + di])

    def get_states(self, out=None, shifted=False):
        """
        Extract the states from all MCMC instances into a single numpy array.

        Args:
            out (np.array): The array to store the states.
            shifted (bool): If True, the states are shifted vertically
                and horizontally by a random amount.

        Returns:
            np.array: An array where each row corresponds to the state of one
                MCMC instance.
        """
        height = self.mcmcs[0].get_height()
        if out is None:
            out = np.empty(
                shape=(len(self.mcmcs), height),
                dtype=np.uint64)
        assert out.shape == (len(self.mcmcs), height)
        for i, mcmc in enumerate(self.mcmcs):
            mcmc._get_state(out=out[i], shifted=shifted)
        return out

    def get_beta_energies(self):
        """
        Get the beta energies of all MCMC instances.

        Returns:
            np.array: An array of beta energies.
        """
        return np.array([mcmc.get_beta_energy() for mcmc in self.mcmcs])

    def get_property(self, name):
        """
        Get the property of all MCMC instances.

        Args:
            name (str): The name of the property to extract.

        Returns:
            np.array: An array of property values.
        """
        return np.array([getattr(mcmc, 'get_' + name)() for mcmc in self.mcmcs])

    def run_isnnn_equilibration(
            self, beta_offset=1.2, num_cooldown_iters=108,
            num_reheat_iters=27, num_inner_steps=8):
        """
        Run equilibration strategy which worked for IsNNN dataset.

        Args:
            beta_offset (float): Increase the beta by this amount during the
                initial cooldown phase.
            num_cooldown_iters (int): Number of iterations to perform during
                the cooldown phase. Each iteration consists of multiple MCMC
                steps and one PT step.
            num_reheat_iters (int): Number of iterations to perform during
                the reheating phase.
            num_inner_steps (int): Number of MCMC steps to perform in each
                iteration.
        """
        beta_true = self.get_property('beta')
        beta_fake = beta_true + beta_offset
        for i in range(len(self)):
            self[i].adjust_some_parameters(beta=beta_fake[i])
        self.step_all(16)
        self.step_pt(di=2)
        for _ in range(num_cooldown_iters):
            self.step_all(num_inner_steps)
            self.step_pt()
        for i in range(len(self)):
            self[i].adjust_some_parameters(beta=beta_true[i])
        for _ in range(num_reheat_iters):
            self.step_all(num_inner_steps)
            self.step_pt()

def estimate_fim_1d(ts, energies, cutoff_t=0.5675):
    """
    Estimate FIM from energies obtained from MCMC simulations.

    Args:
        ts: The temperatures at which the energies were obtained.
            Assumed to be sorted in ascending order.
        energies: The energies obtained at the temperatures ts.
            2D np.ndarray of shape (len(ts), *).
        cutoff_t: The cutoff temperature: FIM is estimated to be 0.0
            for temperatures below cutoff_t.
    """
    assert np.all(ts[1:] > ts[:-1])
    cutoff_i = np.searchsorted(ts, cutoff_t)
    energies = energies[cutoff_i:]
    ts = ts[cutoff_i:]
    dbeta = 1 / ts[1:] - 1 / ts[:-1]
    # z_ratio0 and z_ratio1 are estimates of Z(t + dt) / Z(t):
    z_ratio0 = np.mean(np.exp((-dbeta[:, None]) * energies[:-1]), axis=1)
    z_ratio1 = 1 / np.mean(np.exp(dbeta[:, None] * energies[1:]), axis=1)
    # z_ratio_05 is the estimate of (Z(t + dt) / Z(t))**0.5:
    z_ratio_05 = ((z_ratio0 * z_ratio1)**(1.0 / 4))[:, None]
    r0 = z_ratio_05 * np.exp((dbeta[:, None]/2) * energies[:-1])
    r1 = z_ratio_05 * np.exp((dbeta[:, None]/2) * energies[1:])
    dts = ts[1:] - ts[:-1]
    fim = (4/dts**2) * (
        np.mean(1 - 2 / (r0 + 1/r0), axis=1)
        + np.mean(1 - 2 / (r1 + 1/r1), axis=1))
    fim = np.concatenate([np.zeros(dtype=np.float64, shape=cutoff_i), fim])
    return fim

def generate_1d_dataset(
        seed, ts=None, num_passes=70, num_samples_per_pass_ts=2,
        num_starting_steps=70, min_outer_steps=30, min_inner_steps=9,
        shuffle=True, do_sample=True, **kwargs):
    """
    Generates a dataset for ClassiFIM.

    Args:
        seed: The seed for random number generation.
        ts: Temperature values for dataset generation. Assumed to be sorted
            in ascending order.
            Default: linspace(0, 4, 1001)[1:].
        num_passes: Each pass is a sweep through all temperatures in ts
            in descending order.
        num_samples_per_pass_ts: Number of samples to generate for each
            pair (pass, t).
        num_starting_steps: Number of steps to perform after randomly
            initializing the state.
        min_outer_steps: Number of steps to perform in the outer loop
            (before the difficulty adjustment).
        min_inner_steps: Number of steps to perform in the inner loop
            (before the difficulty adjustment).
        shuffle: If True, shuffle the samples.
        do_sample: If False, skip the sampling (useful when only the FIM
            is needed).
        **kwargs: The keyword arguments for the IsingMCMC2D constructor.
    """
    if ts is None:
        ts = np.linspace(0, 4, 1001)[1:]
    prng = classifim.utils.DeterministicPrng(seed)
    mcmc = IsingMCMC2D(seed=prng.get_int64_seed("IsingMCMC2D"), **kwargs)
    samples_per_ts = num_passes * num_samples_per_pass_ts
    height = mcmc.get_height()
    if do_sample:
        samples = np.empty(
            shape=(len(ts), samples_per_ts, height), dtype=np.uint64)
        # Check manually outside of the loop, so that we can use _get_state:
        assert samples[0, 0, :].flags['C_CONTIGUOUS']
    num_energies_per_pass_ts = (
        min_outer_steps + num_samples_per_pass_ts * min_inner_steps)
    energies = np.empty(
        shape=(len(ts), num_passes * num_energies_per_pass_ts), dtype=np.int32)
    e_i = 0
    for pass_j in range(num_passes):
        pass_offset = pass_j * num_samples_per_pass_ts
        pass_offset_e = pass_j * num_energies_per_pass_ts
        mcmc.reset()
        mcmc.adjust_parameters(beta=1 / ts[-1])
        mcmc.step(num_starting_steps)
        for t_i, t in enumerate(ts[::-1]):
            e_i = pass_offset_e
            mcmc.adjust_parameters(beta=1 / t)
            difficulty_factor = IsingMCMC2D.difficulty_factor(t)
            mcmc.step_combined_ef(
                n_steps=int(min_outer_steps * difficulty_factor),
                energies_out=energies[t_i, e_i:e_i + min_outer_steps],
                flip=False)
            e_i += min_outer_steps
            for j in range(num_samples_per_pass_ts):
                mcmc.step_combined_ef(
                    n_steps=int(min_inner_steps * difficulty_factor),
                    energies_out=energies[t_i, e_i:e_i + min_inner_steps],
                    flip=True)
                e_i += min_inner_steps
                if do_sample:
                    # We manually checked preconditions:
                    # * type, dtype, shape: see initialization above.
                    # * C-contiguous: see assert above.
                    mcmc._get_state(
                        out=samples[t_i, pass_offset + j, :],
                        shifted=True)
    assert e_i == energies.shape[1]

    energies = energies[::-1]
    res = {
        "seed": seed,
        "shuffled": shuffle,
        "width": mcmc.get_width(),
        "height": height,
        "_ts": ts,
        "_energies": energies,
        "_fim": estimate_fim_1d(ts, energies),
    }
    if do_sample:
        ts1 = np.repeat(ts, samples_per_ts)
        samples = samples[::-1].reshape(-1, height)
        if shuffle:
            rng = np.random.Generator(np.random.PCG64(
                prng.get_int64_seed("shuffle")))
            ii = rng.permutation(len(ts1))
            res["_ii"] = ii
            ts1 = ts1[ii]
            samples = samples[ii, :]
        res["ts"] = ts1
        res["samples"] = samples
    return res

def isnnn_lambdas_to_params(lambda0, lambda1):
    j_nn = lambda0
    j_nnn = 1.0 - lambda0
    return {
        "beta": 0.4 / lambda1,
        "jh": -j_nn,
        "jv": -j_nn,
        "jp": -j_nnn,
        "jm": -j_nnn,
        "h": 0.0}

def isnnn_obs_to_energies(lambdas, obss):
    """
    Compute scaled energies of IsNNN.

    H = +J_nn \sum_NN ZZ + J_NNN \sum_NNN ZZZ + const

    const is implementation-defined and may depend on lambdas

    Args:
        lambdas: (n, 2) tensor with lambda0 and lambda1 values.
        obss: (n, num_obs_types) tensor with the values of observables
            num_obs_types is 2 for IsNNN.

    Returns:
        Energies scaled to T=1, i.e. values of beta * H.
    """
    n = max(lambdas.shape[0], obss.shape[0])
    assert lambdas.shape == (n, 2) or lambdas.shape == (1, 2), (
        f"{lambdas.shape} != ({n}, 2)")
    assert obss.shape == (n, 2) or obss.shape == (1, 2), (
        f"{obss.shape} != ({n}, 2)")
    beta = 0.4 / lambdas[:, 1]
    j_nn = lambdas[:, 0] * beta
    j_nnn = (1.0 - lambdas[:, 0]) * beta
    sum_nn = -2 * obss[:, 0]
    sum_nnn = -2 * obss[:, 1]
    return j_nn * sum_nn + j_nnn * sum_nnn

def isnnn_estimate_fim(
        data=None, lambda0s=None, lambda1s=None,
        obss=None, scaling_resolution=None):
    """
    Estimate fisher information metric from observations for IsNNN.

    Args:
        data: dict to extract other arguments from.
        lambda0s: values of lambda0 (J_NN / (J_NN + J_NNN))
        lambda1s: values of lambda1 (T / 2.5)
        obss: np.ndarray of shape (len(lambda0s), len(lambda1s), n, n_types),
            where n_types = 2 (NN and NNN).
        scaling_resolution: inverse lattice spacing
            to be used in FIM scaling.

    Returns:
        FIM in the same format as classifim.bench.fidelity.compute_2d_fim
    """
    assert data is not None or all(
            v is not None for v in [lambda0s, lambda1s, obss])
    if obss is None:
        obss = np.moveaxis(data["obss"], 0, 2)
        n_lambda0s, n_lambda1s, n0, n1, n_types = obss.shape
        n = n0 * n1
        obss = obss.reshape((n_lambda0s, n_lambda1s, n, n_types))
    else:
        n_lambda0s, n_lambda1s, n, n_types = obss.shape
    if lambda0s is None:
        lambda0s = data["lambda0s"]
    if lambda1s is None:
        lambda1s = data["lambda1s"]
    assert (n_lambda0s,) == lambda0s.shape
    assert (n_lambda1s,) == lambda1s.shape
    n_lambdas = (n_lambda0s, n_lambda1s)
    if scaling_resolution is None:
        scaling_resolution = data.get("scaling_resolution")
        if scaling_resolution is None:
            s0 = (n_lambda0s - 1) / (lambda0s[-1] - lambda0s[0])
            s1 = (n_lambda1s - 1) / (lambda1s[-1] - lambda1s[0])
            assert s0 == s1
            assert s0 > 0
            scaling_resolution = s0
    res = {
        "lambda0": [],
        "lambda1": [],
        "dir": [],
        "fim": []}
    cur_lambdas = [
        np.tile(lambda0s[:, None, None], (1, n_lambda1s, n)),
        np.tile(lambda1s[None, :, None], (n_lambda0s, 1, n))]
    cur_lambdas = np.hstack([cl.ravel()[:, None] for cl in cur_lambdas])
    same_energies = isnnn_obs_to_energies(cur_lambdas, obss.reshape(-1, n_types))
    same_energies = same_energies.reshape((n_lambda0s, n_lambda1s, n))
    same_offsets = np.mean(same_energies, axis=2)
    same_ptildes = np.exp(-(same_energies - same_offsets[:, :, None]))
    for direction, v in classifim.bench.fidelity.FIDELITY_DIRECTIONS_2D.items():
        # b = a + v
        # a corresponds to lambda_slices[0]; b corresponds to lambda_slices[1]
        lambda_slices = []
        cur_obsss = []
        cur_lambdass = []
        cur_n_lambdas = np.array(n_lambdas) - np.abs(v)
        for i in range(2):
            lambda_slices.append([
                slice(
                    max((1 - 2 * i) * v[j], 0),
                    n_lambdas[j] - max((2 * i - 1) * v[j], 0))
                for j in range(2)])
            for j in range(2):
                s = lambda_slices[-1][j]
                assert s.stop - s.start == n_lambdas[j] - abs(v[j]), (
                    f"{i=}, {j=}, {n_lambdas[j]}, v[j]={v[j]}, "
                    f"{max((1 - 2 * i) * v[j], 0)}, {max((2 * i - 1) * v[j], 0)}")
            cur_obss = obss[lambda_slices[i][0], lambda_slices[i][1], :, :]
            try:
                cur_obss = cur_obss.reshape(
                    (np.prod(cur_n_lambdas) * n, n_types))
            except ValueError:
                print(f"dir={direction}, {v=}, {lambda_slices[i]=}, {cur_obss.shape=} != "
                    + f"{(cur_n_lambdas[0], cur_n_lambdas[1], n, n_types)}")
                raise
            cur_obsss.append(cur_obss)
            cur_lambdas = [
                    np.tile(
                        lambda0s[lambda_slices[i][0], None, None],
                        (1, cur_n_lambdas[1], n)),
                    np.tile(
                        lambda1s[None, lambda_slices[i][1], None],
                        (cur_n_lambdas[0], 1, n))]
            cur_lambdass.append(np.hstack([cl.ravel()[:, None] for cl in cur_lambdas]))
        cur_ptildes = [[None, None] for i in range(2)]
        for i in range(2):
            cur_ptildes[i][i] = same_ptildes[lambda_slices[i][0], lambda_slices[i][1], :]
            other_energies = isnnn_obs_to_energies(cur_lambdass[i], cur_obsss[1 - i])
            other_energies = other_energies.reshape((cur_n_lambdas[0], cur_n_lambdas[1], n))
            other_offsets = same_offsets[lambda_slices[i][0], lambda_slices[i][1], None]
            cur_ptildes[i][1 - i] = np.exp(-(other_energies - other_offsets))
        cur_ptildes = np.array(cur_ptildes)
        # Formula (L5)
        sqrt_z_ratio = (np.mean(cur_ptildes[1, 0] / cur_ptildes[0, 0], axis=2) / (
                np.mean(cur_ptildes[0, 1] / cur_ptildes[1, 1], axis=2)))**(1/4)
        rx = (cur_ptildes[0] / cur_ptildes[1])**0.5 * sqrt_z_ratio[None, :, :, None]
        fim = 8 * np.mean(1 - 2 / (rx + 1 / rx), axis=(0, 3)) * scaling_resolution**2
        cur_lambdas_shape = (cur_n_lambdas[0], cur_n_lambdas[1], n, 2)
        mid_lambda = (
            cur_lambdass[0].reshape(cur_lambdas_shape)[:, :, 0]
            + cur_lambdass[1].reshape(cur_lambdas_shape)[:, :, 0]) / 2
        res["lambda0"].append(mid_lambda[:, :, 0].ravel())
        res["lambda1"].append(mid_lambda[:, :, 1].ravel())
        res["fim"].append(fim.ravel())
        res["dir"].append(np.tile(direction, np.prod(cur_n_lambdas)).ravel())
        assert res["lambda0"][-1].shape == res["fim"][-1].shape, (
            f"{res['lambda0'][-1].shape} != {res['fim'][-1].shape}")
    for key in list(res.keys()):
        res[key] = np.concatenate(res[key])
    return pd.DataFrame(res)

def _isnnn_generate_data_single_pass(
        mcmc, n_steps_per_sample, samples, obss, equil_args=None):
    mcmc.run_isnnn_equilibration(**(equil_args or {}))
    num_samples = samples.shape[1]
    num_obss = obss.shape[1]
    obss_pos = 0
    assert samples.shape == (
        len(mcmc), num_samples, mcmc[0].get_height())
    assert obss.shape == (
        len(mcmc), num_obss, 2)
    assert samples[0, 0, :].flags['C_CONTIGUOUS']
    assert samples.dtype == np.uint64
    for j in range(num_samples):
        if j > 0:
            mcmc.step_pt()
        next_obss_pos = (j + 1) * obss.shape[1] // num_samples
        for i, mcmci in enumerate(mcmc):
            mcmci.step_combined_2of(
                n_steps=n_steps_per_sample,
                observables_out=obss[i, obss_pos:next_obss_pos, :],
                flip=True)
            mcmci._get_state(samples[i, j, :], shifted=True)
        obss_pos = next_obss_pos
    assert obss_pos == obss.shape[1]

def isnnn_generate_data(
        seed,
        lambda0s=None,
        lambda1s=None,
        num_passes=70,
        num_samples_per_pass_lambda=2,
        num_steps_per_sample=32,
        lambdas_to_params=isnnn_lambdas_to_params,
        width=20,
        height=20,
        num_threads=None,
        use_tqdm=False):
    """
    Generates Ising NNN for ClassiFIM.

    Args:
        seed: The seed for random number generation.
        lambda0s: Values for lambda0 (J_NN / (J_NN + J_NNN))
        lambda1s: Values for lambda1 (temperature = 2.5 * lambda1)
        num_passes: Each pass is a sweep through all lambda1s in ts
            in descending order.
        num_samples_per_pass_lambda: Number of samples to generate for each
            pair (pass, t).
        num_steps_per_sample: Number of MCMC steps to perform before each sample.
        lambdas_to_params: Function that converts lambda0, lambda1 to parameters
        width: The width of the 2D lattice.
        height: The height of the 2D lattice.
        num_threads: Number of threads to use.

    Returns:
        dict with keys:
            samples: np.ndarray with type np.uint64 and shape
                (num_passes, len(lambda0s), len(lambda1s),
                num_samples_per_pass_lambda, height)
            obss: np.ndarray with type np.int32 and shape
                (num_passes, len(lambda0s), len(lambda1s),
                num_samples_per_pass_lambda * 32, 2)
    """
    if lambda0s is None:
        lambda0s = (np.arange(64) + 1) / 64
    assert len(lambda0s.shape) == 1
    if lambda1s is None:
        lambda1s = (np.arange(64) + 1) / 64
    assert len(lambda1s.shape) == 1
    prng = classifim.utils.DeterministicPrng(seed)
    samples = np.empty(
        shape=(num_passes, len(lambda0s), len(lambda1s),
               num_samples_per_pass_lambda, height),
        dtype=np.uint64)
    obss = np.empty(
        shape=(num_passes, len(lambda0s), len(lambda1s),
               num_samples_per_pass_lambda * num_steps_per_sample, 2),
        dtype=np.int32)
    def run(pass_i, lambda0_i, progress=None):
        lambda0 = lambda0s[lambda0_i]
        mcmc = PTArray([
            IsingNNNMCMC(
                width=width, height=height,
                seed=prng.get_int64_seed((
                    "IsingNNNMCMC", pass_i, lambda0_i, lambda1_i)),
                **lambdas_to_params(lambda0, lambda1))
            for lambda1_i, lambda1 in enumerate(lambda1s)])
        equil_args = {}
        if 40 <= lambda0_i <= 43:
            equil_args["num_cooldown_iters"] = [
                216, 432, 432, 216][lambda0_i - 40]
        _isnnn_generate_data_single_pass(
            mcmc,
            num_steps_per_sample,
            samples[pass_i, lambda0_i, :, :, :],
            obss[pass_i, lambda0_i, :, :, :],
            equil_args=equil_args)
        if progress is not None:
            progress.update(1)
        return True
    runs = itertools.product(range(num_passes), range(len(lambda0s)))
    if num_threads is None:
        if use_tqdm:
            runs = tqdm(runs, total=num_passes * len(lambda0s))
        for run_i in runs:
            run(*run_i)
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads) as executor:
            if use_tqdm:
                progress = tqdm(total=num_passes * len(lambda0s))
            else:
                progress = None
            res = executor.map(lambda r: run(*r, progress=progress), list(runs))
            for r in res:
                assert r
            if use_tqdm:
                progress.close()
    return {
        "lambda0s": lambda0s,
        "lambda1s": lambda1s,
        "samples": samples,
        "obss": obss,
        "width": width,
        "height": height}


def isnnn_generate_datasets(
        seed, data, train_filename=None, test_filename=None):
    """
    Generate IsNNN datasets for ClassiFIM.

    Args:
        seed: The seed for random number generation.
        data: The MCMC data generated by isnnn_generate_data.
        train_filename: Save the training dataset to this file.
        test_filename: Save the test dataset to this file.

    Returns:
        d_train, d_test: The training and test datasets.
    """
    prng = classifim.utils.DeterministicPrng(seed)
    samples = data["samples"]
    lambda0s = data["lambda0s"]
    lambda1s = data["lambda1s"]
    num_passes, num_lambda0s, num_lambda1s, num_samples0, height = samples.shape
    # num_samples0 = num_samples per (pass, lambda0, lambda1)
    assert height == data["height"]
    assert lambda0s.shape == (num_lambda0s,)
    assert lambda1s.shape == (num_lambda1s,)
    width = data["width"]
    # num_samples1 = num_samples per (lambda0, lambda1)
    num_samples1 = num_passes * num_samples0
    num_samples = num_lambda0s * num_lambda1s * num_samples1
    samples = np.moveaxis(samples, 0, 2).reshape((num_samples, height))
    samples = classifim.io.samples2d_to_bytes(samples, width)
    lambda0s = np.tile(lambda0s[:, None, None], (1, num_lambda1s, num_samples1))
    lambda1s = np.tile(lambda1s[None, :, None], (num_lambda0s, 1, num_samples1))
    d_all = {
        "lambda0s": lambda0s,
        "lambda1s": lambda1s,
        "samples": samples,
        "width": width,
        "height": height,
        "seed": seed}
    np_rng = np.random.default_rng(prng.get_int64_seed("shufle_test"))
    idx = np_rng.permutation(num_samples)
    for k in ['lambda0s', 'lambda1s', 'samples']:
        d_all[k] = d_all[k].reshape((num_samples,))[idx]
    d_train, d_test = classifim.io.split_train_test(
            d_all,
            test_size=0.1,
            seed=prng.get_int64_seed("split_test"),
            scalar_keys=["width", "height", "seed"])
    if train_filename is not None:
        classifim_gen.io.isnnn_dataset_save(d_train, train_filename)
    if test_filename is not None:
        classifim_gen.io.isnnn_dataset_save(d_test, test_filename)
    return d_train, d_test

def ising400_tc_summary(
        ts, fim, n_spins, tc_theory=2 / np.log(1 + 2**0.5), print_=True):
    """
    Compute the critical temperature and its uncertainty from FIM values.

    Args:
        ts: The temperatures at which the FIM values were computed.
        fim: The FIM values.
        n_spins: Number of spins (used for printing the summary).
        tc_theory: The theoretical critical temperature
            (used for printing the summary).
        print_: If True, print the summary.

    Returns:
        Dict with keys:
            tc: The critical temperature.
            tc_std: The standard deviation of the critical temperature.
    """
    res = {}
    res["tc"] = tc = np.array([
        ts[np.argmax(fim[i])] for i in range(fim.shape[0])])
    res["tc_mean"] = np.mean(tc)
    res["tc_std"] = np.std(tc)
    res["tcinv_mean"] = np.mean(1 / tc)
    res["tcinv_std"] = np.std(1 / tc)
    res["betac"] = betac = np.array([
        1 / ts[np.argmax(fim[i] * ts**4)] for i in range(fim.shape[0])])
    res["betac_mean"] = np.mean(betac)
    res["betac_std"] = np.std(betac)
    res["betacinv_mean"] = np.mean(1 / betac)
    res["betacinv_std"] = np.std(1 / betac)
    res["num_ds"] = len(tc)
    if print_:
        if n_spins is not None:
            print(f"N = {n_spins}")
        print(f"Tc = {res['tc_mean']:.3f} ± {res['tc_std']:.3f}; "
            f"1/βc = {res['betacinv_mean']:.3f} ± {res['betacinv_std']:.3f}"
            f" (theory: {tc_theory:.3f})")
        print(f"1/Tc = {res['tcinv_mean']:.4f} ± {res['tcinv_std']:.4f}; "
            f"βc = {res['betac_mean']:.4f} ± {res['betac_std']:.4f}"
            f" (theory: {1 / tc_theory:.4f}; num_ds={len(tc)})")
    return res


def ising400_plot_2bands(ax, datasets, set_ylim=True, ymax=None,
        label=None, print_summary=True, n_sites=None, **kwargs):
    """
    A plot of mean and std band of FIM value and the value of its maximum
    for Ising400 and similar datasets.

    The FIM value is scaled by num_sites.

    Args:
        ax: The matplotlib axis to plot on.
        datasets: List of datasets, each of them is a dict (e.g. returned by
            generate_1d_dataset) with keys "packed_zs", "_ts", "_fim", "width".
        set_ylim: If True, set the y-axis limits to [0, max(fim_ub)].
        ymax: Value to be used for y-axis limit and vertical lines
            (computed if None).
        label: Label for the plot.
        print_summary: If True, print the summary about Tc.
        kwargs: Additional arguments for ax.fill_between and ax.plot.

    Returns:
        Dict with the following keys:
            ymax: The maximum value of the upper bound on FIM computed
                by this function.
    """
    if isinstance(datasets, dict):
        datasets = list(datasets.values())
    ts = np.array([ds["_ts"] for ds in datasets])
    ts_mid = (ts[:, 1:] + ts[:, :-1]) / 2
    assert all(np.array_equal(
        ts_mid[0], ts_mid[i]) for i in range(1, len(ts_mid)))
    ts_mid = ts_mid[0]
    if n_sites is None:
        n_sites = np.array([
            ds["width"] * ds["packed_zs"].shape[1] for ds in datasets])
    elif isinstance(n_sites, int):
        n_sites = np.full(len(datasets), n_sites)
    fim = np.array([ds["_fim"] for ds in datasets])
    fim = fim / n_sites[:, None]
    fim_mean = np.mean(fim, axis=0)
    fim_std = np.std(fim, axis=0)
    fim_lb = fim_mean - fim_std
    fim_ub = fim_mean + fim_std
    computed_ymax = max(fim_ub)
    if ymax is None:
        ymax = computed_ymax
    ax.fill_between(ts_mid, fim_lb, fim_ub, alpha=0.3, **kwargs)
    plot_kwargs = dict(kwargs)
    if label is not None:
        plot_kwargs["label"] = label
    ax.plot(ts_mid, fim_mean, **plot_kwargs)

    tc = np.array([ts_mid[np.argmax(fim[i, :])] for i in range(fim.shape[0])])
    tc_mean = np.mean(tc)
    tc_std = np.std(tc)
    ax.fill_betweenx(
        [0, ymax], tc_mean - tc_std, tc_mean + tc_std, alpha=0.3, **kwargs)
    ax.axvline(tc_mean, **kwargs)
    if set_ylim:
        ax.set_ylim([0, ymax])
    ax.set_xlim((np.min(ts), np.max(ts)))
    ax.set_xlabel("T")
    ax.set_ylabel("FIM / num_sites")
    res = {"ymax": ymax}
    if print_summary:
        assert np.all(n_sites == n_sites[0])
        res.update(
            ising400_tc_summary(ts_mid, fim, n_sites[0], print_=True))
    return res

