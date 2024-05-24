"""
Functions and classes relevant to Frustrated Ising Ladder Hamiltonian
on a 24-site lattice. The lattice is based on 12-site 2D lattice
in hubbard_hamiltonian.py. It has 2 copies of the 12-site lattice
stacked on top of each other.
"""

import classifim_gen.hamiltonian as hamiltonian
import ctypes
import functools
import math
import numpy as np
import os
import scipy.sparse
import sys
from tqdm import tqdm
from classifim.bits import reverse_bit_pairs_uint32, countbits32, \
        roll_left, extract_every_second_bit_uint32, \
        spread_to_every_second_bit_uint32

class Fil1DFamily(hamiltonian.LinearHamiltonianFamily):
    """
    Frustrated Ising Ladder on 1D-ish lattice with periodic boundary conditions.

    Note that this class can be used to describe Frustrated Ising Ladder
    on 2D lattices if the cell of that lattice forms a finite cyclic group
    (e.g. 12-site lattice follows this rule).

    Below comments assume 12-site lattice is used. We have to deal with
    3 lattices here:
    - Top lattice: 12-site lattice (24 edges).
    - Bottom lattice: same as top lattice.
    - Qubit lattice: lattice with 24 sites, consisting of the
    top and bottom rows joined together.

    Classical state of the system is encoded as 24-bit bitstring z s.t.
    Z_{Bj} = (-1) ** ((z >> 2j) & 1) = 1 - 2 * ((z >> 2j) & 1)
    Z_{Tj} = (-1) ** ((z >> (2j + 1)) & 1) = 1 - 2 * ((z >> (2j + 1)) & 1)

    Examples (nsites=16):
    * z = 0xaaaaaaaa: top row is all 1s (ZT = -1), bottom row is all 0s (ZB = 1).
    * z = 0x55555555: top row is all 0s (ZT = 1), bottom row is all 1s (ZB = -1).

    Attributes
    ----------
    nsites : int
        Number of sites of the top lattice.
        The number of classical states is 4**nsites.
    edge_dirs : list[int]
        Directions for edges in the top lattice. The set of corresponding
        pairs (j, k) is referred to as NN (nearest neighbors):
        (j, k) in NN iff (j - k) % nsites in edge_dirs.
    z_to_vi : array of length 4**nsites.
        For each classical state z, z_to_vi[z] is the corresponding
        'vi', i.e. the index of its equivalence class with respect to
        translations and reflections.
    vi_to_zs : array of shape (num_vis, 2 * nsites)
        For each 'vi', vi_to_zs[vi] is the set of corresponding classical
        states (padded with uint32(-1)).
        For each 'vi' an associated state is
        $|S[vi]> = \sum_{z in vi_to_zs[vi]} |z> / sqrt(vi_count[vi])$.
    vi_count : array of length num_vis. The number of classical
        states in each equivalence class.
    op_x_sparse : scipy.sparse.csr_matrix describing
        $-\sum_j (X_{B{j}} + X_{Tj})$.
    op_kterms_vec : array of shape (num_vis, ) describing
        $$\sum_{(j,k) in NN} (-Z_{B{j}} Z_{B{k}} + Z_{T{j}} Z_{T{k}}) * 2 / len(NN(j))
          + \sum_{j} (-Z_{B{j}} Z_{T{j}} - Z_{T{j}}).$$
        That is, the term in the Hamiltonian which should be multiplied by
        $k \cdot s$.
    op_uterms_vec : array of shape (num_vis, ) describing $\sum_{j} Z_{B{j}} / 2$.
        That is, the term in the Hamiltonian which should be multiplied by
        $u \cdot s$.
    """
    PARAM_NAMES = ("gamma", "su", "sk")

    def __init__(self, nsites=12, edge_dirs=(1, 3, 9, 11)):
        self.nsites = nsites
        self.edge_dirs = edge_dirs
        self.init_lookups()
        self.init_ops()

    def init_lookups(self):
        all_zs = np.arange(2**(2 * self.nsites), dtype=np.uint32)
        all_zs_transformed = [all_zs]
        cur_zs = all_zs
        mask = (1 << (2 * self.nsites)) - 1
        # Add translations:
        assert self.nsites <= 15
        for i in range(1, self.nsites):
            cur_zs = cur_zs << 2
            cur_zs = (cur_zs & mask) + (cur_zs >> (2 * self.nsites))
            all_zs_transformed.append(cur_zs)
        all_zs_transformed = np.array(all_zs_transformed).T
        # Add reflections:
        all_zs_reflected = reverse_bit_pairs_uint32(all_zs_transformed)
        all_zs_reflected >>= 32 - 2 * self.nsites
        # 1.5Gb for nsites=12:
        all_zs_transformed = np.hstack((
            all_zs_transformed,
            all_zs_reflected))
        assert all_zs_transformed.shape == (2**(2 * self.nsites), 2 * self.nsites), (
            f"{all_zs_transformed.shape=} != (2**{2 * self.nsites}, {2 * self.nsites})")
        # Representative is the smallest element in the equivalence class:
        all_zs_smallest = np.min(all_zs_transformed, axis=1)
        # vi is the index of the equivalence class:
        unique_zs, vi_to_z, self.z_to_vi, self.vi_count = np.unique(
            all_zs_smallest, return_index=True, return_inverse=True,
            return_counts=True)
        assert self.z_to_vi.shape == (2**(2 * self.nsites),), (
            f"{self.z_to_vi.shape=} != 2**{2 * self.nsites}")
        raw_vi_to_zs = np.sort(all_zs_transformed[vi_to_z, :], axis=1)
        # At this point every row of raw_vi_to_zs is a set of equivalent
        # classical states with multiplicities. We need to remove
        # multiplicities and pad with -1s. Note though that
        # the height of vi_to_zs is large (of the order of
        # 2**(2 * nsites) / (2 * nsites)). Hence, the above should be done
        # in a vectorized way.
        vi_to_zs = np.full(raw_vi_to_zs.shape, -1, dtype=np.uint32)
        cur = raw_vi_to_zs[:, 0]
        vi_to_zs[:, 0] = cur
        pos = np.full(vi_to_zs.shape[0], 0, dtype=np.int32)
        assert raw_vi_to_zs.shape[1] == 2 * self.nsites
        for i in range(1, raw_vi_to_zs.shape[1]):
            prev = cur
            cur = raw_vi_to_zs[:, i]
            pos += (cur != prev)
            vi_to_zs[:, pos] = cur
        self.vi_to_zs = vi_to_zs

    def init_ops(self):
        self.init_x_op()
        self.init_z_ops()

    def init_x_op(self):
        """
        Init self.op_x_sparse
        """
        # Here we use symmetry of H_X = -sum_l(X_l) to
        # simplify the computation by only applying H_X
        # to the first representative of each equivalence class
        # |z> of |S[vi]>:
        # <S[vi']|sum_l(X_l)|S[vi]>
        # = vi_count[vi]**(+0.5) * <S[vi']|sum_l(X_l)|z>
        vi_from = np.arange(self.vi_to_zs.shape[0])
        zs_from = self.vi_to_zs[:, 0]
        x_total = scipy.sparse.csr_matrix(
                (self.space_dim, self.space_dim),
                dtype=np.float64)
        for j in range(2 * self.nsites):
            zs_to = zs_from ^ (1 << j)
            vi_to = self.z_to_vi[zs_to]
            count_to = self.vi_count[vi_to].astype(np.float64)
            coeff = -np.sqrt(self.vi_count / count_to)
            x_total += scipy.sparse.csr_matrix(
                (coeff, (vi_to, vi_from)),
                shape=(self.space_dim, self.space_dim))
        self.op_x_sparse = x_total

    def init_z_ops(self):
        """
        Init self.op_kterms_vec and self.op_uterms_vec
        """
        zs = self.vi_to_zs[:, 0]
        assert zs.dtype == np.uint32
        zbs = (zs & 0x55555555).astype(np.uint64)
        zts = ((zs >> 1) & 0x55555555).astype(np.uint64)
        # -\sum_{j} Z_{T{j}}:
        term_zt = self.nsites - countbits32(zts) * 2
        term_zb = self.nsites - countbits32(zbs) * 2
        # -\sum_{j} Z_{B{j}} Z_{T{j}}:
        term_zbzt = self.nsites - countbits32(zbs ^ zts) * 2
        term_horizontal = 0
        for delta in self.edge_dirs:
            zts_shifted = roll_left(zts, 2 * delta, 2 * self.nsites)
            zbs_shifted = roll_left(zbs, 2 * delta, 2 * self.nsites)
            term_horizontal += self.nsites - countbits32(zts ^ zts_shifted) * 2
            term_horizontal -= self.nsites - countbits32(zbs ^ zbs_shifted) * 2
        term_horizontal *= 2 / len(self.edge_dirs)
        self.op_kterms_vec = term_horizontal - term_zt - term_zbzt
        self.op_uterms_vec = term_zb / 2

    @property
    def space_dim(self):
        return self.vi_to_zs.shape[0]

    @property
    def h_terms(self):
        return [
            self.op_x_sparse.dot,
            self.op_uterms_vec.dot,
            self.op_kterms_vec.dot]

    def apply_h(self, params_vec: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Apply the Hamiltonian to a vector."""
        gamma, su, sk = params_vec
        return (gamma * self.op_x_sparse.dot(vec) +
            su * self.op_uterms_vec.dot(vec) +
            sk * self.op_kterms_vec.dot(vec))

def divisors_gen(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n // i)
    for divisor in reversed(large_divisors):
        yield divisor

@functools.lru_cache(maxsize=24)
def bracelet_count(length, symmetry_type="", num_bead_types=4):
    """
    Recursive function to count the number of different bracelets
    of given length and symmetry type.

    Args:
        length (int): length of the bracelet
        symmetry_type (str): symmetry type of the bracelet; one of
          - "": no reflection symmetry
          - "S": reflection symmetry about axis going through a bead
          - "D": reflection symmetry about axis going between two beads
          Special treatments:
          - if length is odd, "S" and "D" have the same meaning
          - if length == 1, symmetry_type is ignored (as both 'S' and 'D'
          are no-op)
          - if length == 2
            * "S" is ignored (as it is no-op)
            * for "D" *_cur values are 0 since (2, "D") is the same as (1, "").
        num_bead_types (int): number of different bead types

    Returns: tuple with the following components:
        - diff_cur: count of different bracelets, not counting bracelets
        with additional symmetries.
        - total_cur: total count of bracelets, not counting bracelets
        with additional symmetries.
        - diff_total: count of different bracelets, including bracelets
        with additional symmetries.
        - total_total: total count of bracelets, including bracelets
        with additional symmetries (e.g. num_bead_types**length when
        symmetry_type == "").
    """
    assert length > 0
    assert symmetry_type in ["", "S", "D"]
    if length <= 2:
        # Base case
        if length == 1:
            return num_bead_types, num_bead_types, num_bead_types, num_bead_types
        # length == 2
        if symmetry_type == "D":
            # 'S' symmetry for length == 2 effectively reduces the length to 1.
            # Hence any bracelet will have additional symmetries.
            return 0, 0, num_bead_types, num_bead_types
        # symmetry_type in ["", "S"]
        total_cur = num_bead_types * (num_bead_types - 1)
        diff_cur = total_cur // 2
        diff_total = diff_cur + num_bead_types
        return total_cur // 2, total_cur, diff_total, num_bead_types**2
    # length > 2
    divisors = list(divisors_gen(length))
    # subsymmetries is a list of tuples
    # (sub_length, sub_symmetry_type, count)
    # Where count is the number of conjugacy classes
    # of the symmetry (sub_length, sub_symmetry_type)
    subsymmetries = []
    for divisor in divisors:
        if divisor < length:
            subsymmetries.append((divisor, symmetry_type, 1))
        if symmetry_type == "" and divisor > 2:
            if divisor % 2 == 0:
                subsymmetries.append((divisor, "S", divisor // 2))
                subsymmetries.append((divisor, "D", divisor // 2))
            else:
                subsymmetries.append((divisor, "S", divisor))
    if symmetry_type == "":
        total_total = num_bead_types**length
    else:
        effective_length = (length + (1 if symmetry_type == "D" else 2)) // 2
        total_total = num_bead_types**effective_length
    total_others = 0
    diff_others = 0
    for sub_length, sub_symmetry_type, count in subsymmetries:
        o_diff_cur, o_total_cur, o_diff_total, o_total_total = bracelet_count(
            sub_length, sub_symmetry_type, num_bead_types)
        total_others += count * o_total_cur
        diff_others += count * o_diff_cur
    total_cur = total_total - total_others
    orbit_size_cur = length * (2 if symmetry_type == "" else 1)
    assert total_cur % orbit_size_cur == 0
    diff_cur = total_cur // orbit_size_cur
    diff_total = diff_cur + diff_others
    return diff_cur, total_cur, diff_total, total_total

class CppFil1DFamily:
    PARAM_NAMES = ("gamma", "su", "sk")
    def __init__(self, nsites, edge_dirs):
        """
        Frustrated Ising Ladder Hamiltonian family with 1D translation symmetry.

        Implemented in C++.

        Args:
            nsites (int): number of sites in each row of the ladder.
            edge_dirs (list of int): directions of the edges: nodes T_{j} and T_{k} are
                connected iff (j - k) % nsites in edge_dirs.
        """
        self.nsites = nsites
        self.edge_dirs = edge_dirs.copy()

        # Assign None first in case of exception:
        self._lib = None
        self._lib = self._get_lib()

        edge_dirs_array = (ctypes.c_int * len(edge_dirs))(*edge_dirs)
        _fill1d_family = self._lib.create_fil1d_family(
                nsites, edge_dirs_array, len(edge_dirs))
        self._fil1d_family = ctypes.c_void_p(_fill1d_family)

    @staticmethod
    def _get_lib():
        extension = '.dll' if sys.platform == 'win32' else '.so'
        lib_path = os.path.join(
            os.path.dirname(__file__),
            'lib',
            'libclassifim_gen' + extension)
        lib = ctypes.CDLL(lib_path)

        # create_fil1d_family
        lib.create_fil1d_family.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.create_fil1d_family.restype = ctypes.c_void_p

        # delete_fil1d_family
        lib.delete_fil1d_family.argtypes = [ctypes.c_void_p]
        lib.delete_fil1d_family.restype = None

        # get_op_kterms
        lib.fil1d_family_get_op_kterms.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.fil1d_family_get_op_kterms.restype = ctypes.POINTER(ctypes.c_double)

        # get_op_uterms
        lib.fil1d_family_get_op_uterms.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.fil1d_family_get_op_uterms.restype = ctypes.POINTER(ctypes.c_double)

        # get_op_x
        lib.fil1d_family_get_op_x.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                       ctypes.POINTER(ctypes.c_int),
                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                       ctypes.POINTER(ctypes.c_int)]
        lib.fil1d_family_get_op_x.restype = None

        # get_z_to_vi
        lib.fil1d_family_get_z_to_vi.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_int)]
        lib.fil1d_family_get_z_to_vi.restype = ctypes.POINTER(ctypes.c_uint32)

        # get_vi_to_z
        lib.fil1d_family_get_vi_to_z.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_int)]
        lib.fil1d_family_get_vi_to_z.restype = ctypes.POINTER(ctypes.c_uint32)

        # get_orbit_sizes
        lib.fil1d_family_get_orbit_sizes.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_int)]
        lib.fil1d_family_get_orbit_sizes.restype = ctypes.POINTER(ctypes.c_uint8)
        return lib

    def __del__(self):
        if self._lib is not None:
            self._lib.delete_fil1d_family(self._fil1d_family)

    @functools.cached_property
    def op_kterms(self):
        size = ctypes.c_int()
        kterms_ptr = self._lib.fil1d_family_get_op_kterms(self._fil1d_family, ctypes.byref(size))
        assert size.value > 0
        # Create a numpy array from the pointer:
        kterms_array = np.ctypeslib.as_array(kterms_ptr, (size.value,))
        # Makes a copy of the numpy array to avoid referencing C++ memory
        return kterms_array.copy()

    @functools.cached_property
    def op_uterms(self):
        size = ctypes.c_int()
        uterms_ptr = self._lib.fil1d_family_get_op_uterms(self._fil1d_family, ctypes.byref(size))
        assert size.value > 0
        uterms_array = np.ctypeslib.as_array(uterms_ptr, (size.value,))
        return uterms_array.copy()

    @functools.cached_property
    def op_x_csr(self):
        row_ptrs = ctypes.POINTER(ctypes.c_int)()
        nrows = ctypes.c_int()
        col_idxs = ctypes.POINTER(ctypes.c_int)()
        data = ctypes.POINTER(ctypes.c_double)()
        nnz = ctypes.c_int()

        self._lib.fil1d_family_get_op_x(
            self._fil1d_family, ctypes.byref(row_ptrs), ctypes.byref(nrows),
            ctypes.byref(col_idxs), ctypes.byref(data), ctypes.byref(nnz))

        row_ptrs_array = np.ctypeslib.as_array(row_ptrs, (nrows.value + 1,))
        col_idxs_array = np.ctypeslib.as_array(col_idxs, (nnz.value,))
        data_array = np.ctypeslib.as_array(data, (nnz.value,))

        csr = scipy.sparse.csr_matrix(
            (data_array, col_idxs_array, row_ptrs_array),
            dtype=np.float64)
        return csr

    @functools.cached_property
    def z_to_vi(self):
        size = ctypes.c_int()
        z_to_vi_ptr = self._lib.fil1d_family_get_z_to_vi(self._fil1d_family, ctypes.byref(size))
        assert size.value > 0
        z_to_vi_array = np.ctypeslib.as_array(z_to_vi_ptr, (size.value,))
        return z_to_vi_array.copy()

    @functools.cached_property
    def vi_to_z(self):
        size = ctypes.c_int()
        vi_to_z_ptr = self._lib.fil1d_family_get_vi_to_z(self._fil1d_family, ctypes.byref(size))
        assert size.value > 0
        vi_to_z_array = np.ctypeslib.as_array(vi_to_z_ptr, (size.value,))
        return vi_to_z_array.copy()

    @functools.cached_property
    def orbit_sizes(self):
        size = ctypes.c_int()
        orbit_sizes_ptr = self._lib.fil1d_family_get_orbit_sizes(self._fil1d_family, ctypes.byref(size))
        assert size.value > 0
        orbit_sizes_array = np.ctypeslib.as_array(orbit_sizes_ptr, (size.value,))
        return orbit_sizes_array.copy()

    @functools.cached_property
    def space_dim(self):
        return self.op_kterms.shape[0]

    def apply_h_fun(self, params_vec):
        """
        Returns a function that applies the Hamiltonian to a vector.

        Args:
            params_vec (np.ndarray): Vector of parameters (gamma, su, sk).
        """
        gamma, su, sk = params_vec
        op_x_csr = gamma * self.op_x_csr
        op_z_vec = su * self.op_uterms + sk * self.op_kterms
        return lambda vec: op_x_csr.dot(vec) + op_z_vec * vec

    def apply_h(self, params_vec, vec: np.ndarray) -> np.ndarray:
        """
        Applies Hamiltonian to a vector.

        Args:
            params_vec (np.ndarray): Vector of parameters (gamma, su, sk).
            vec (np.ndarray): Vector to apply Hamiltonian to. Should have
                shape=(space_dim,).
        """
        gamma, su, sk = params_vec
        return (
            gamma * self.op_x_csr.dot(vec)
            + (su * self.op_uterms + sk * self.op_kterms) * vec)

    @functools.cached_property
    def vi_to_abs_mtp(self) -> np.ndarray:
        """
        Absolute staggered magnetization of top row for each vi.

        Scaled by nsites (= the number of sites in the top row).
        Value is in [0, 1]:
        0 = all spins are aligned
        1 = spins alternate up/down

        Returns:
            np.ndarray: Array of shape (space_dim,).
        """
        zs = self.vi_to_z
        TOP_ROW_MASK = 0xaaaaaaaa
        TOP_ROW_STAGGERED = 0x88888888
        zs_top_staggered = (zs & TOP_ROW_MASK) ^ TOP_ROW_STAGGERED
        mtp_raw = classifim_gen.bits.countbits32(zs_top_staggered)
        return np.abs(1.0 - mtp_raw * (2.0 / self.nsites))

    @functools.cached_property
    def vi_to_mb(self) -> np.ndarray:
        """
        Magnetization of the top row for each vi.

        Scaled by nsites (= the number of sites in the bottom row).
        Value is in [-1, 1].
        -1 = all zs are 1 (Z = -1)
        1 = all zs are 0 (Z = 1).

        Returns:
            np.ndarray: Array of shape (space_dim,).
        """
        zs = self.vi_to_z
        BOTTOM_ROW_MASK = 0x55555555
        zs_bottom = zs & BOTTOM_ROW_MASK
        mb_raw = classifim_gen.bits.countbits32(zs_bottom)
        return 1.0 - mb_raw * (2.0 / self.nsites)

    def randomly_transform_z(self, zs: np.ndarray, seed=42) -> np.ndarray:
        """
        Apply a random Hamiltonian symmetry to each z in zs.
        """
        assert zs.dtype == np.uint32
        rng = np.random.default_rng(seed)
        rolls = rng.integers(0, self.nsites, size=zs.shape, dtype=np.uint32)
        flips = rng.integers(0, 2, size=zs.shape)
        res = roll_left(zs, 2 * rolls, 2 * self.nsites)
        res[flips == 1] = reverse_bit_pairs_uint32(res[flips == 1]) >> (32 - 2 * self.nsites)
        assert res.data is not zs.data
        assert res.dtype == np.uint32
        return res

    def repack_z_for_bschifc(self, zs: np.ndarray) -> np.ndarray:
        """
        Repacks zs into the format expected by bschifc.

        Input zs represents a configuration of the Fil1D model.
        Bit Tj of zs is (zs & (1 << (1 + 2 * j))) > 0.
        Bit Bj of zs is (zs & (1 << (2 * j))) > 0.

        Output zs (zs_out) is repacked for bschifc.
        Bit Tj of zs_out is (zs_out & (1 << (self.nsites + j))) > 0.
        Bit Bj of zs_out is (zs_out & (1 << j)) > 0.

        In other words, input bits are in the order 0bTBTBTB...TB,
        where T denotes top row bits and B denotes bottom row bits.
        Output bits are in the order 0bTT...TTBB...BB.
        """
        assert zs.dtype == np.uint32
        return (
            extract_every_second_bit_uint32(zs)
            | (extract_every_second_bit_uint32(zs >> 1) << self.nsites))

    def unrepack_z_for_bschifc(self, zs: np.ndarray) -> np.ndarray:
        """
        Inverse of repack_z_for_bschifc.
        """
        assert zs.dtype == np.uint32
        zs_bot = zs & ((1 << self.nsites) - 1)
        zs_top = zs >> self.nsites
        return (
            spread_to_every_second_bit_uint32(zs_bot)
            | (spread_to_every_second_bit_uint32(zs_top) << 1))

class Fil24ParamConversions:
    @staticmethod
    def lambdas_to_params(lambda0, lambda1):
        """
        Converts lambda0 and lambda1 to gamma, su, sk for Fil1DFamily.

        Args:
            lambda0 (float): s
            lambda1 (float): u
        """
        s = lambda0
        u = lambda1
        k = 1.0
        gamma = 1 - s
        su = s * u
        sk = s * k
        return gamma, su, sk

    @staticmethod
    def params_to_lambdas(gamma, su, sk):
        """
        Converts gamma, su, sk to lambda0 and lambda1 for Fil1DFamily.
        """
        s = sk / (sk + gamma)
        u = su / sk if sk != 0 else 0.0
        return s, u

def _pack_lambdas_as_key(lambdas, resolution=64):
    """
    Repacks lambdas into a single integer key.
    """
    assert lambdas.shape[-1] == 2
    lambdais = (lambdas * resolution + 0.5).astype(np.int32)
    assert np.all(lambdas == lambdais / resolution)
    assert np.all(0 <= lambdais)
    assert np.all(lambdais < resolution)
    return lambdais[..., 1] * resolution + lambdais[..., 0]

def compute_best_possible_xe(
        dump_npz, zs_to_vi_f, lambdas_to_params_f, probs_cache, resolution=64):
    """
    Compute best possible cross-entropy error for a dump of test data.

    Args:
        dump_npz: dump of test data, dict or NpzFile with the following keys:
          - lambda0s
          - dlambdas
          - zs
          - labels
        zs_to_vi_f: function to map zs to vi
        lambdas_to_params_f: function to map lambdas to parameters
          (e.g. gamma, su, sk for FIL24).
        probs_cache: classifim_gen.ground_state_cache with probabilities.

    Returns:
        float: best possible cross-entropy error.
    """
    num_samples, = dump_npz["zs"].shape
    assert dump_npz["lambda0s"].shape == (num_samples, 2)
    lambdas = [
        dump_npz["lambda0s"] + pm * dump_npz["dlambdas"] / 2
        for pm in (1, -1)]
    lambdas_stack = np.concatenate(lambdas)
    lambdais_stack = _pack_lambdas_as_key(lambdas_stack, resolution)
    zs_stack = np.tile(dump_npz["zs"].astype(np.uint32), 2)

    sort_ii = np.argsort(lambdais_stack)
    unsort_ii = np.zeros_like(sort_ii)
    unsort_ii[sort_ii] = np.arange(len(sort_ii))

    lambdais_stack_sorted = lambdais_stack[sort_ii]
    zs_stack_sorted = zs_stack[sort_ii]
    cutoffs = np.where(
        lambdais_stack_sorted[1:] != lambdais_stack_sorted[:-1])[0] + 1
    assert len(cutoffs) == resolution**2 - 1
    cutoffs = np.concatenate(([0], cutoffs, [len(lambdais_stack)]))
    vi_stack_sorted = zs_to_vi_f(zs_stack_sorted)

    probs_stack_sorted = []
    for cutoff_j in tqdm(range(resolution**2)):
        lambda0i = cutoff_j % resolution
        lambda1i = cutoff_j // resolution
        jmin = cutoffs[cutoff_j]
        jmax = cutoffs[cutoff_j + 1]
        cur_vi = vi_stack_sorted[jmin:jmax]
        cur_lambdas = (l / resolution for l in (lambda0i, lambda1i))
        params_vec = lambdas_to_params_f(*cur_lambdas)
        npz = probs_cache.get_ground_state(tuple(params_vec))
        probs_stack_sorted.append(npz["probs"][cur_vi])

    probs_stack = np.concatenate(probs_stack_sorted)[unsort_ii]
    assert probs_stack.shape == (2 * num_samples,)
    probs_p = probs_stack[:num_samples]
    probs_m = probs_stack[num_samples:]
    probs_predicted = probs_p / (probs_p + probs_m)
    probs_predicted[dump_npz["labels"] == 0] = (
        1 - probs_predicted[dump_npz["labels"] == 0])
    return -np.mean(np.log(probs_predicted))



