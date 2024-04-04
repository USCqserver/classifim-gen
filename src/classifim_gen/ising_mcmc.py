import classifim.utils
import ctypes
import numpy as np
import os
import sys
import functools

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

        lib.ising_mcmc2D_base_get_energy0.argtypes = [ctypes.c_void_p]
        lib.ising_mcmc2D_base_get_energy0.restype = ctypes.c_int

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
        lib.ising_nnn_mcmc_get_beta.restype = ctypes.c_int

        lib.ising_nnn_mcmc_get_h.argtypes = [
            ctypes.c_void_p,  # const classifim_gen::IsingNNNMCMC *mcmc
        ]
        lib.ising_nnn_mcmc_get_h.restype = ctypes.c_double

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

    def reset(self):
        self._lib.ising_mcmc2D_base_reset(self._mcmc)

    def step(self, n_steps=1):
        self._lib.ising_mcmc2D_base_step(self._mcmc, ctypes.c_int(n_steps))

    def get_width(self):
        return self._lib.ising_mcmc2D_base_get_width(self._mcmc)

    def get_height(self):
        return self._lib.ising_mcmc2D_base_get_height(self._mcmc)

    def get_magnetization(self):
        return self._lib.ising_mcmc2D_base_get_magnetization(self._mcmc)

    def get_energy0(self):
        """
        Returns the energy of the current state, assuming h=0.
        """
        return self._lib.ising_mcmc2D_base_get_energy0(self._mcmc)

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
    # create_ising_nnn_mcmc(std::uint64_t seed, int width, int height, double beta,
    #                   double jh, double jv, double jp, double jm, double h) {
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

    def adjust_parameters(
            self, beta=ISING_2D_BETA_CRITICAL,
            jh=1.0, jv=1.0, jp=0.0, jm=0.0, h=0.0):
        self._lib.ising_nnn_mcmc_adjust_parameters(
            self._mcmc,
            ctypes.c_double(beta),
            ctypes.c_double(jh),
            ctypes.c_double(jv),
            ctypes.c_double(jp),
            ctypes.c_double(jm),
            ctypes.c_double(h))

    def step_flip(self):
        self._lib.ising_nnn_mcmc_step_flip(self._mcmc)

    def get_beta(self):
        return self._lib.ising_nnn_mcmc_get_beta(self._mcmc)

    def get_h(self):
        return self._lib.ising_nnn_mcmc_get_h(self._mcmc)

def estimate_fim(ts, energies, cutoff_t=0.5675):
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
        shuffle=True, **kwargs):
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
        **kwargs: The keyword arguments for the IsingMCMC2D constructor.
    """
    if ts is None:
        ts = np.linspace(0, 4, 1001)[1:]
    prng = classifim.utils.DeterministicPrng(seed)
    mcmc = IsingMCMC2D(seed=prng.get_int64_seed("IsingMCMC2D"), **kwargs)
    samples_per_ts = num_passes * num_samples_per_pass_ts
    height = mcmc.get_height()
    samples = np.empty(
        shape=(len(ts), samples_per_ts, height), dtype=np.uint64)
    min_outer_steps = 30
    min_inner_steps = 9
    num_energies_per_pass_ts = (
        min_outer_steps + num_samples_per_pass_ts * min_inner_steps)
    energies = np.empty(
        shape=(len(ts), num_passes * num_energies_per_pass_ts), dtype=np.int32)
    # Check manually outside of the loop, so that we can use _get_state:
    assert samples[0, 0, :].flags['C_CONTIGUOUS']
    e_i = 0
    for pass_j in range(num_passes):
        pass_offset = pass_j * num_samples_per_pass_ts
        pass_offset_e = pass_j * num_energies_per_pass_ts
        mcmc.reset()
        mcmc.adjust_parameters(beta=1 / ts[-1])
        mcmc.step(70)
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
                # We manually checked preconditions:
                # * type, dtype, shape: see initialization above.
                # * C-contiguous: see assert above.
                mcmc._get_state(
                    out=samples[t_i, pass_offset + j, :],
                    shifted=True)
    assert e_i == energies.shape[1]

    ts1 = np.repeat(ts, samples_per_ts)
    samples = samples[::-1].reshape(-1, height)
    if shuffle:
        rng = np.random.Generator(np.random.PCG64(
            prng.get_int64_seed("shuffle")))
        ii = rng.permutation(len(ts1))
        ts1 = ts1[ii]
        samples = samples[ii, :]
    energies = energies[::-1]
    return {
        "ts": ts1,
        "packed_zs": samples,
        "seed": seed,
        "shuffled": shuffle,
        "width": mcmc.get_width(),
        "_ts": ts,
        "_ii": ii,
        "_energies": energies,
        "_fim": estimate_fim(ts, energies),
    }
