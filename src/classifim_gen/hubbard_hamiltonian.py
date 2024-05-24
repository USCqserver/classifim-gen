"""
Functions and classes relevant to the Hubbard Hamiltonian on a 12-site lattice.
"""

import classifim.io
import classifim_gen.hamiltonian as hamiltonian
import classifim_gen.io
import functools
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.special
from classifim.bits import countbits16

@functools.lru_cache(maxsize=32)
def compute_choices(n: int, k: int) -> np.ndarray:
    """
    Returns an array of all possible bitstrings of length n with k bits set.
    """
    assert n <= 15
    if k <= 1 or n - k <= 1:
        if k == 0:
            return np.array([0])
        if k == 1:
            return 2 ** np.arange(n)
        return np.flip(
            np.array(2 ** n - 1)
            - compute_choices(n, n - k))
    n1 = n // 2
    n2 = n - n1
    k1max = min(k, n1)
    k2max = min(k, n2)
    k1min = k - k2max
    res = []
    for k1 in range(k1min, k1max + 1):
        choices1 = compute_choices(n1, k1)
        choices2 = compute_choices(n2, k - k1)
        res.append((choices1[:, np.newaxis] * 2 ** n2 + choices2).reshape(-1))
    return np.concatenate(res)


class Hubbard1DFamily(hamiltonian.LinearHamiltonianFamily):
    """
    Hubbard Hamiltonian family on a 1D-ish lattice with periodic boundary conditions.

    Note that this class can be used to describe Hubbard Hamiltonian on 2D lattices
    if the cell of that lattice forms a finite cyclic group
    (e.g. 12-site lattice follows this rule).

    Attributes
    ----------
    nsites : int
        Number of sites on the lattice.
    nfilled : int
        Amount of fermions of each spin.
        TODO: Generalize to allow for different number of up and down fermions.
    nn_edge_dirs : list[int]
        Directions for nearest-neighbor hopping terms.
    nnn_edge_dirs : list[int]
        Directions for next-nearest-neighbor hopping terms.
    dtype_int : type
        Data type for integer arrays.
    dtype_float : type
        Data type for floating point arrays.
    zs_per_spin : array of length choose(nsites, nfilled)
        All bitstrings of length nsites with nfilled bits set.
        bit #j of zs_per_spin[i] is 1 iff the creation operator c_{j,sigma}
        is present in the term corresponding to the bitstring zs_per_spin[i].
    zs_parity_per_spin : array of length choose(nsites, nfilled)
        Bit #j of zs_parity_per_spin[i] is set the number of fermion operators
        from nsites-1 till j inclusive mod 2 for bitstring zs_per_spin[i].
    zs_per_spin_lookup : array of length 2**nsites,
        zs_per_spin[zs_per_spin_lookup[z]] = z whenever z is present in
        zs_per_spin; otherwise zs_per_spin_lookup[z] = -1.
    h_int_precomp : 1D array of the length len(zs_per_spin)**2
        Precomputed values for h_int s.t. h_int(v) = u * self.h_int_precomp * v
    h_nn_precomp : list
        List [lplus, lminus], each of which is a list of (viold, vinew) s.t.
        h_nn(v) can be computed by operations of the form
            res[:, vinew] += v2[:, viold]
            res[vinew, :] += v2[viold, :]
        where v2 = v.reshape((len(zs_per_spin), len(zs_per_spin)))
    h_nnn_precomp : list
        Same as h_nn_precomp, but for next-nearest-neighbor jumps.
    """
    PARAM_NAMES = ("u", "tnn", "tnnn")

    def __init__(
            self, dtype_int=np.int16, dtype_float=np.float32, nsites=12, nfilled=6,
            nn_edge_dirs=(1, 9, 3, 11), nnn_edge_dirs=(2, 10, 4, 8), xp=np
    ):
        self.nsites = nsites
        self.nfilled = nfilled
        self.nn_edge_dirs = nn_edge_dirs
        self.nnn_edge_dirs = nnn_edge_dirs
        self.dtype_int = dtype_int
        self.dtype_float = dtype_float
        self._set_xp(xp)
        self.zs_per_spin = compute_choices(self.nsites, self.nfilled)
        assert len(self.zs_per_spin) == scipy.special.comb(self.nsites, self.nfilled, exact=True)
        self._space_dim = len(self.zs_per_spin)**2
        # zs parity: 0b111001 -> 0b101110
        self.zs_parity_per_spin = self.compute_zsp(self.zs_per_spin, self.nsites)
        self.zs_per_spin_lookup = self.compute_lookup(
            self.zs_per_spin,
            length=2 ** self.nsites,
            default_value=-1)
        self.h_int_precomp = self.xp_asarray(self.compute_hint_precomp())
        self.h_nn_precomp = self.compute_hkin_precomp(
            self.nn_edge_dirs)
        self.h_nnn_precomp = self.compute_hkin_precomp(
            self.nnn_edge_dirs)

    def _set_xp(self, xp):
        """
        Sets up the numpy-like library to use.
        """
        self.xp = xp
        if hasattr(xp, "asarray"):
            self.xp_asarray = xp.asarray
        else:
            self.xp_asarray = xp.array
        if hasattr(xp, "asnumpy"):
            self.xp_asnumpy = xp.asnumpy
        else:
            self.xp_asnumpy = np.array

    def z_to_vi(self, z_down, z_up):
        """Converts a bitstring to a state index.

        :param z_down: bitstring for down fermions (of length nsites)
        :param z_up: bitstring for up fermions (of length nsites)
        :return: state index (between 0 and choose(nsites, nfilled)**2)
        """
        vi_down = self.zs_per_spin_lookup[z_down].astype(np.uint32)
        vi_up = self.zs_per_spin_lookup[z_up].astype(np.uint32)
        return vi_down * len(self.zs_per_spin) + vi_up

    @functools.cached_property
    def vi_to_z(self):
        """Converts a state index to a bitstring.

        Note:
            z_down = self.zs_per_spin[vi // len(self.zs_per_spin)]
            z_up = self.zs_per_spin[vi % len(self.zs_per_spin)]
            z = z_down * 2**nsites + z_up
        """
        z_up, z_down = [
            a.ravel().astype(np.uint32)
            for a in np.meshgrid(self.zs_per_spin, self.zs_per_spin)]
        return (z_down << self.nsites) | z_up

    @staticmethod
    def compute_zsp(zs, nsites):
        """
        Computes a bitstring counting the number of 1s to the left or at a given
        location mod 2.

        E.g. 111001 -> 101110
        """
        res = np.full(zs.shape, 0, dtype=zs.dtype)
        for i in range(nsites):
            zs_bit = (zs >> i) & 1
            res = res ^ ((zs_bit << (i + 1)) - zs_bit)
        return res

    def compute_lookup(self, target, length, default_value):
        """Computes lookup table for target.

        Args:
          target: 1D np.ndarray to lookup into.
            elements of target should be integers in range(length)
          length: length of the lookup array.
          default_value: value of the lookup array for
            elements not present in target.

        Returns:
          1D np.ndarray lookup of length `length`
          s.t. target[lookup[z]] == z whenever z is in target,
          otherwise lookup[z] = default_value.
        """
        assert length <= 2 ** 15
        lookup = np.full(length, fill_value=default_value, dtype=self.dtype_int)
        lookup[target] = np.arange(len(target), dtype=self.dtype_int)
        return lookup

    def compute_hint_precomp(self):
        xor_array = self.zs_per_spin[:, np.newaxis] ^ self.zs_per_spin[np.newaxis, :]
        xor_array = xor_array.reshape(-1)
        res = 0.25 * (self.nsites - 2 * countbits16(xor_array)).astype(
            self.dtype_float)
        return self.xp_asarray(res)

    @staticmethod
    def _hkin_precomp_remove_empty(precomps):
        return [(a, b) for (a, b) in precomps if len(a) > 0]

    def compute_hkin_precomp(self, edge_dirs):
        res_plus = []
        res_minus = []
        for j in range(self.nsites):
            for edge_dir in edge_dirs:
                jold = (j + edge_dir) % self.nsites
                precomp_plus, precomp_minus = self.compute_jump_precomp(j, jold)
                res_plus.append(precomp_plus)
                res_minus.append(precomp_minus)
        return (
            self._hkin_precomp_remove_empty(res_plus),
            self._hkin_precomp_remove_empty(res_minus))

    def compute_jump_precomp(self, j, jold):
        """
        Computes (vcoef, vinew, viold) corresponding to the jump from jold to j.
        Then encodes it as (vinew_plus, viold_plus), (vinew_minus, viold_minus)

        Let
          * jump = c_{j}^\\dagger c_{jold}
          * w = w0 + jump(v)
        Then w can be computed as follows:
          w = w0.copy()
          w[vinew] += vcoef * v[viold]
        or as:
          w = w0.copy()
          w[vinew] += v[viold_plus]
          w[vinew] -= v[viold_minus]

        If z is a bitstring corresponding to vinew[k], then
          * z[j] = 1
          * z[jold] = 0
          * vcoef[k] is
            (-1)^{#bits set in j strictly between positions j and jold}
        """
        assert j != jold

        # Compute (-1)^{#bits set from max(j, jold) excl. till min(j, jold) incl:
        vcoef = 1 - 2 * ((
                (self.zs_parity_per_spin >> j) ^ (self.zs_parity_per_spin >> jold)
            ) & 1)

        # j > jold: z[min(j, jold)] = z[jold] = 0, vcoef is correct.
        # j < jold: z[min(j, jold)] = z[j] = 1, hence we are off by 1.
        if j < jold:
            vcoef = -vcoef

        has_j = (self.zs_per_spin >> j) & 1
        has_jold = (self.zs_per_spin >> jold) & 1
        valid_move = np.logical_and(has_j, np.logical_not(has_jold))
        viold = self.zs_per_spin_lookup[self.zs_per_spin ^ ((1 << j) ^ (1 << jold))]
        vinew = np.arange(len(self.zs_per_spin), dtype=self.dtype_int)

        vcoef = vcoef[valid_move]
        viold = viold[valid_move]
        vinew = vinew[valid_move]

        assert np.all(viold >= 0)
        vinew_plus = self.xp_asarray(vinew[vcoef > 0], dtype=self.dtype_int)
        vinew_minus = self.xp_asarray(vinew[vcoef < 0], dtype=self.dtype_int)
        viold_plus = self.xp_asarray(viold[vcoef > 0], dtype=self.dtype_int)
        viold_minus = self.xp_asarray(viold[vcoef < 0], dtype=self.dtype_int)

        return (vinew_plus, viold_plus), (vinew_minus, viold_minus)

    def apply_hint(self, v):
        return self.h_int_precomp * v

    def apply_hnn(self, v):
        return self.apply_hkin(self.h_nn_precomp, v)

    def apply_hnnn(self, v):
        return self.apply_hkin(self.h_nnn_precomp, v)

    @staticmethod
    def _apply_half_hkin(jumps_vcoef_vi, v, res):
        for vinew, viold in jumps_vcoef_vi[0]:
            res[vinew, :] += v[viold, :]
        for vinew, viold in jumps_vcoef_vi[1]:
            res[vinew, :] -= v[viold, :]
            # Note: we don't do
            # res[:, vinew] -= v[:, viold]
            # here for efficiency reasons
            # (this operation is not SIMD or cache friendly)
            # hence 'half' in the function name.

    def apply_hkin(self, jumps_vcoef_vi, v):
        len_per_spin = len(self.zs_per_spin)
        shape_2d = (len_per_spin, len_per_spin)
        v = v.reshape(shape_2d)
        res = self.xp.zeros(shape_2d, dtype=self.dtype_float)
        self._apply_half_hkin(jumps_vcoef_vi, v, res)
        v2 = v.T.reshape(-1).reshape(shape_2d)
        res2 = self.xp.zeros(shape_2d, dtype=self.dtype_float)
        self._apply_half_hkin(jumps_vcoef_vi, v2, res2)
        res += res2.T
        return res.reshape(-1)

    def _apply_half_hkin_v2(self, jumps_vcoef_vi, v, shape_2d):
        res = self.xp.zeros(shape_2d, dtype=self.dtype_float)
        for vinew, viold in jumps_vcoef_vi[0]:
            res[vinew, :] += v[viold, :]
        for vinew, viold in jumps_vcoef_vi[1]:
            res[vinew, :] -= v[viold, :]
        return res

    def _apply_half_hkins(self, jumps_data, coefs, v, shape_2d):
        res = self._apply_half_hkin_v2(jumps_data[0], v, shape_2d) * coefs[0]
        for i in range(1, len(coefs)):
            res += self._apply_half_hkin_v2(jumps_data[i], v, shape_2d) * coefs[i]
        return res

    def apply_hkins(self, jumps_data, coefs, v):
        assert len(jumps_data) == len(coefs)
        len_per_spin = len(self.zs_per_spin)
        shape_2d = (len_per_spin, len_per_spin)
        v = v.reshape(shape_2d)
        res = self._apply_half_hkins(jumps_data, coefs, v, shape_2d)
        v2 = v.T.reshape(-1).reshape(shape_2d)
        res2 = self._apply_half_hkins(jumps_data, coefs, v2, shape_2d)
        res += res2.T
        return res.reshape(-1)

    def apply_hnn_hnnn(self, v, cnn, cnnn):
        return self.apply_hkins(
            (self.h_nn_precomp, self.h_nnn_precomp), (cnn, cnnn), v)

    def apply_h(self, params, v):
        v = self.xp_asarray(v, dtype=self.dtype_float)
        u, tnn, tnnn = params
        return self.xp_asnumpy(
            u * self.apply_hint(v) + self.apply_hnn_hnnn(v, -tnn, -tnnn))

    def apply_h_fun(self, params_vec):
        """
        Returns a function that applies the Hamiltonian to a vector.

        Args:
            params_vec (np.ndarray): Vector of parameters (u, tnn, tnnn).
        """
        u, tnn, tnnn = params_vec
        h_int_precomp = u * self.h_int_precomp
        return lambda v: self.xp_asnumpy(
            h_int_precomp * v + self.apply_hnn_hnnn(v, -tnn, -tnnn))

    @property
    def h_terms(self):
        """
        Return the list of terms in the Hamiltonian.

        The Hamiltonian is a linear combination of terms. This function
        returns the list of corresponding terms (each of them being a function).
        """

        return [
            lambda v: self.xp_asnumpy(self.apply_hint(self.xp_asarray(v))),
            lambda v: self.xp_asnumpy(-self.apply_hnn(self.xp_asarray(v))),
            lambda v: self.xp_asnumpy(-self.apply_hnnn(self.xp_asarray(v)))]

    @property
    def space_dim(self) -> int:
        return self._space_dim

class HubbardParamsConversions:
    """
    Convert between (u, t, t') and (lambda0, lambda1).
    """
    def __init__(self, hubbard_family):
        self.n_sites = hubbard_family.nsites
        self.hint_norm_per_site = 0.25
        self.hkin_norm_per_site = 8 * hubbard_family.nfilled / self.n_sites

    def lambdas_to_params(self, lambda0, lambda1):
        """
        Compute

        We compute u, t, t' by solving:
        t / (u + t) = lambda0
        t' / (u + t') = lambda1
        u + t + t' = 1
        Then we normalize each term by dividing it by the norm upper bound of
        the corresponding Hamiltonian term per site.

        Args:
            lambda0: tnn0 / (u0 + tnn0), between 0 and 1
            lambda1: tnnn0 / (u0 + tnnn0), between 0 and 1
            Both lambda0 and lambda1 can't be 1 simultaneously

        Returns: u, tnn, tnnn
        """
        lambda_det = 1 - lambda0 * lambda1
        u = (1 - lambda0) * (1 - lambda1) / lambda_det
        tnn = lambda0 * (1 - lambda1) / lambda_det
        tnnn = lambda1 * (1 - lambda0) / lambda_det
        u /= self.hint_norm_per_site
        tnn /= self.hkin_norm_per_site
        tnnn /= self.hkin_norm_per_site
        return u, tnn, tnnn

    def params_to_lambdas(self, u, tnn, tnnn):
        """
        Compute lambda0, lambda1 from params (u, tnn, tnnn)

        params could be floats or numpy arrays of the same size.
        """
        u0 = u * self.hint_norm_per_site
        tnn0 = tnn * self.hkin_norm_per_site
        tnnn0 = tnnn * self.hkin_norm_per_site
        lambda0 = tnn0 / (u0 + tnn0)
        lambda1 = tnnn0 / (u0 + tnnn0)
        return lambda0, lambda1

def hubbard12_dataset_save(dataset, filename, num_lambdas=2):
    """
    Save the dataset in the format of Hubbard12 dataset.

    Args:
        dataset (dict): Dictionary with the following keys:
            - 'lambda0s' (np.ndarray): Array of floats.
            - 'lambda1s' (np.ndarray): Array of floats.
            - 'samples' (np.ndarray): Array of np.uint32s
        filename (str): Path to the output file.
        num_lambdas: Number of lambda columns in the dataset.
    """
    # Sanity check num_lambdas:
    assert 0 < num_lambdas
    for i in range(num_lambdas):
        assert f'lambda{i}s' in dataset
    assert f'lambda{i+1}s' not in dataset

    if filename.endswith(".parquet"):
        lambda_fields = [
            pa.field(f"lambda{i}", pa.float32(), nullable=False)
            for i in range(num_lambdas)]
        schema = pa.schema(lambda_fields + [
                pa.field("sample", pa.int32(), nullable=False)])

        # Remove last 's' from keys:
        # Naming convention for features stored in npz is to use plural form
        # (e.g. lambda0s) to distinguish them from metadata (e.g. width).
        # On the other hand, storing
        # metadata in parquet columns is not supported (all parquet columns
        # have the same number of rows), so column name is feature name
        # (singular form).
        dataset = {
            k: dataset[k + 's']
            for k in ([f'lambda{i}' for i in range(num_lambdas)] + ['sample'])}
        assert dataset['sample'].dtype in (
            np.uint32, np.int32, np.int64, np.uint64)
        if dataset['sample'].dtype != np.int32:
            assert np.all(dataset['sample'] <= 2**31 - 1)
            dataset['sample'] = dataset['sample'].astype(np.int32)
        table = pa.Table.from_pydict(dataset, schema=schema)
        pq.write_table(table, filename)
    elif filename.endswith(".npz"):
        np.savez_compressed(filename, **dataset)
    else:
        raise ValueError(f"Unknown extension: {filename}")

def convert_dataset_to_hf(
        seed, data, train_filename=None, test_filename=None,
        num_lambdas=2, samples_column_name="zs", samples_dtype=np.uint32,
        scalar_keys=None):
    """
    Converts Hubbard12 dataset from .npz to format compatible with HuggingFace.

    Also works for FIL24 dataset because it uses the same dataset format.

    Arguments after `test_filename` are optional and introduced to allow
    for code reuse (converting other similar datasets).

    Args:
        seed (int): Seed for the random number generator.
        data: either
            - str: Path to the input .npz file containing the dataset
                with keys specified below.
            - dict: Dictionary with the following keys:
                - 'lambdas' (np.ndarray): Array of floats.
                - 'zs' (np.ndarray): Array of uint8s or int8s.
                - [optional] scalar keys in [
                    'size_per_lambda', 'seed', 'sample_type'].
        train_filename (str): Path to the output train file.
        test_filename (str): Path to the output test file.
    """
    if isinstance(data, str):
        filename_in = data
        with np.load(filename_in) as npz:
            data = dict(npz)
    if scalar_keys is None:
        scalar_keys = ["size_per_lambda", "num_sites", "num_bits"]
    assert isinstance(scalar_keys, list) or scalar_keys == 'all'
    samples = data[samples_column_name]
    num_samples, = samples.shape
    lambdas = data["lambdas"]
    assert lambdas.shape == (num_samples, num_lambdas)
    if samples_dtype is not None:
        assert samples.dtype == samples_dtype
    d_all = {
        **{f"lambda{i}s": lambdas[:, i] for i in range(num_lambdas)},
        "samples": samples}
    if scalar_keys == 'all':
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    v = v.item()
                else:
                    continue
            d_all[k] = v
    else:
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
        hubbard12_dataset_save(d_train, train_filename, num_lambdas=num_lambdas)
    if test_filename is not None:
        hubbard12_dataset_save(d_test, test_filename, num_lambdas=num_lambdas)
    return d_train, d_test

