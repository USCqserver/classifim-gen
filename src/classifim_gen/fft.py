"""
Fermionic Fourier Transform

This module contains the functions for the fermionic Fourier transform,
i.e. the transform of a vector $\ket{\psi}$ of $2^L$ amplitudes
(complex numbers) to a vector $\ket{\phi}$ of the same length
defined as follows.
$$\ket{\psi} = \sum_{z=0}^{2^L-1} \psi_z \ket{z}, \tag{1}$$
where
$$\ket{z} = \prod_{j=L-1,\dots,0} (c^\dagger_j)^{z_j} \ket{0}, \tag{2}$$
where $c^\dagger_j$ are the fermionic creation operators.

Fermionic Fourier transform is equivalent to replacing each $c^\dagger_j$ with
$$\sum_{k=0}^{2^L-1} \omega^{-jk} c^\dagger_k,$$
and then writing the result in the form (1).
"""

import numpy as np
import functools
from . import bits

def quspin_conversion(psi):
    """
    This reverses the order elements of psi.

    This operation is equal to its inverse and is used to convert
    to the format used by QuSpin.

    The index ordering is tested in test_quspin.TestFermionQuSpin.test_index.
    """
    return psi[::-1]

def apply_creation(j, psi):
    """
    Apply the creation operator $c^\dagger_j$ to the vector $\ket{\psi}$.

    Args:
        j: The index of the creation operator.
        psi: The vector $\ket{\psi}$ with $2^L$ amplitudes.

    Returns:
        The vector $c^\dagger_j \ket{\psi}$.
    """
    L = int(np.log2(len(psi)) + 0.5)
    assert 0 <= j
    assert j < L
    assert 1 <= L
    assert L <= 31

    # Bitstrings with 0 at j-th position:
    j_mask = 1 << j
    j_right_mask = j_mask - 1
    even_zs = np.arange(1 << (L-1), dtype=np.uint32)
    even_zs = (even_zs & j_right_mask) | ((even_zs & ~j_right_mask) << 1)

    zs_left = even_zs >> (j + 1)
    sign = 1 - ((bits.countbits32(zs_left) & 1) << 1).astype(np.int32)
    res = np.zeros_like(psi)
    res[even_zs + j_mask] = sign * psi[even_zs]
    return res

def apply_annihilation(j, psi):
    """
    Apply the annihilation operator $c_j$ to the vector $\ket{\psi}$.

    Args:
        j: The index of the annihilation operator.
        psi: The vector $\ket{\psi}$ with $2^L$ amplitudes.

    Returns:
        The vector $c_j \ket{\psi}$.
    """
    L = int(np.log2(len(psi)) + 0.5)
    assert 0 <= j
    assert j < L
    assert 1 <= L
    assert L <= 31

    # Bitstrings with 0 at j-th position:
    j_mask = 1 << j
    j_right_mask = j_mask - 1
    even_zs = np.arange(1 << (L-1), dtype=np.uint32)
    even_zs = (even_zs & j_right_mask) | ((even_zs & ~j_right_mask) << 1)

    zs_left = even_zs >> (j + 1)
    sign = 1 - ((bits.countbits32(zs_left) & 1) << 1).astype(np.int32)
    res = np.zeros_like(psi)
    res[even_zs] = sign * psi[even_zs + j_mask]
    return res

def apply_occupation_number(j, psi):
    """
    Apply the occupation number operator $n_j$ to the vector $\ket{\psi}$.

    $$ n_j = c^\dagger_j c_j $$

    Args:
        j: The index of the occupation operator.
        psi: The vector $\ket{\psi}$ with $2^L$ amplitudes.

    Returns:
        The vector $n_j \ket{\psi}$.
    """
    L = int(np.log2(len(psi)) + 0.5)
    assert 0 <= j
    assert j < L
    assert 1 <= L
    assert L <= 31

    n = ((np.arange(1 << L, dtype=np.uint32) >> j) & 1).astype(psi.dtype)
    return psi * n

def apply_z(j, psi):
    """
    Apply $z_j = n_j - 1/2$ to the vector $\ket{\psi}$.

    See `apply_occupation_number` for the definition of $n_j$.

    Args:
        j: The index of the occupation operator.
        psi: The vector $\ket{\psi}$ with $2^L$ amplitudes.

    Returns:
        The vector $z_j \ket{\psi}$.
    """

    L = int(np.log2(len(psi)) + 0.5)
    assert 0 <= j
    assert j < L
    assert 1 <= L
    assert L <= 31

    n = ((np.arange(1 << L, dtype=np.uint32) >> j) & 1).astype(psi.dtype)
    return psi * (n - 0.5)

def givens_decompose(a):
    """
    Decompose a unitary matrix $a$ into a product of 2x2 unitaries.

    This uses the idea from the Givens decomposition.

    Args:
        a: A unitary matrix.

    Returns: a list of length L * (L-1) / 2, where L is a.shape[0].
        Each element is a tuple (g, (i, j)), where g is a 2x2 unitary matrix
        and i, j are the indices of the rows on which g acts.
    """
    L = a.shape[0]
    assert a.shape == (L, L)
    assert L >= 1, f"givens_decompose({L=}): expected L >= 1"
    if L == 1:
        assert np.allclose(a, 1), f"{a=}"
        return []
    givens_rotations = []
    a = a.copy()
    for j in range(L - 1):
        for i in range(L - 1, j, -1):
            a0 = a[i - 1, j]
            a1 = a[i, j]

            # Compute the Givens rotation that will zero a[i, j]
            r = np.hypot(np.abs(a0), np.abs(a1))
            c = a0 / r
            s = a1 / r

            # 2x2 Givens matrix
            G = np.array([[c, -np.conj(s)], [s, np.conj(c)]], dtype=a.dtype)

            # Apply Givens rotation to the matrix a
            a[i - 1:i + 1, :] = np.conj(G.T) @ a[i - 1:i + 1, :]
            assert abs(a[i, j]) < 1e-10

            # Record the Givens rotation
            givens_rotations.append((G, (i - 1, i)))

    # Ensure a[L-1, L-1] == 1
    G, (i0, i1) = givens_rotations.pop()
    assert i0 == L - 2
    assert i1 == L - 1
    a_ii = a[i1, i1]
    G[:, 1] *= a_ii
    a[i1, :] *= np.conj(a_ii)
    givens_rotations.append((G, (i0, i1)))

    assert np.allclose(a, np.eye(L)), f"{L=}, {a=}"
    return givens_rotations

@functools.cache
def _factorize2(n):
    """
    For an integer n, factor n into a product p * n1, where p is prime.

    If n is prime, return (n,).
    """
    factors = []
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return (i, n // i)
    return (n,)

def dft_matrix(L):
    """
    Compute the DFT matrix of size L.

    Args:
        L: The size of the DFT matrix.

    Returns:
        The DFT matrix of size L.
    """
    idx = np.arange(L)
    f = L**(-0.5) * np.exp((-2j * np.pi / L) * idx[:, None] * idx[None, :])
    return f

def _repeat_rotations(rotations, L1):
    """
    Repeat rotations for decomposition of a larger matrix.

    We assume the matrix index is of the form l2 * L1 + l1,
    where rotations should act on index l2.
    """
    res = []
    for l1 in range(L1):
        for g, (i0, i1) in rotations:
            res.append((g, (i0 * L1 + l1, i1 * L1 + l1)))
    return res

def _tile_rotations(rotations, L1, L2):
    """
    Tile rotations for decomposition of a larger matrix.

    We assume the matrix index is of the form l2 * L1 + l1,
    where rotations should act on index l1.
    """
    res = []
    for l2 in range(L2):
        for g, (i0, i1) in rotations:
            res.append((g, (l2 * L1 + i0, l2 * L1 + i1)))
    return res

def _tile_permutation(perm, L2):
    """
    Tile permutation for decomposition of a larger matrix.

    We assume the matrix index is of the form l2 * L1 + l1,
    and the input permutation permutes indices l1 in 0, ..., L1-1.
    The output permutation should permute indices 0, ..., L2*L1-1.
    """
    if perm is None:
        return None
    L1 = len(perm)
    return np.tile(perm, L2) + np.repeat(np.arange(L2) * L1, L1)

def _push_diag_to_rotations(d, rotations):
    """
    Multiply the rotations by a diagonal matrix on the left.
    """
    d = d.copy()
    res = []
    for g, (i0, i1) in rotations:
        g = d[[i0, i1], None] * g # copy
        d[[i0, i1]] = 1
        res.append((g, (i0, i1)))
    return res

@functools.cache
def _fermionic_ft_plan(L):
    """
    Compute the FermionicFT plan for the fermionic Fourier transform.

    To perform FermionicFT we need to decompose FFT matrix $F$ (L by L)
    into a product
    F = P @ u[k-1] @ ... @ u[0],
    where P is a permutation matrix and u[j] are unitary matrices acting
    non-trivially on only 2 rows. Each u[j] is encoded as a tuple
    (g, (i0, i1)), where g is a 2x2 unitary matrix and i0, i1 are the
    indices of the rows on which g acts.

    Args:
        L: The number of qubits.

    Returns: tuple (perm, rotations), where
        perm is None or a permutation of range(L),
        rotations is a list of tuples (g, (i0, i1)).
    """
    if L == 1:
        return None, []
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    if L == 2:
        return None, [(h, (0, 1))]
    if L == 3:
        h3 = np.array([[1, -1j], [1, 1j]], dtype=np.complex128) / np.sqrt(2)
        u3i = np.array([[3**(-0.5), (2/3)**0.5], [(2/3)**0.5, -3**(-0.5)]],
                       dtype=np.double)
        return None, [(h3, (1, 2)), (u3i, (0, 1)), (h, (1, 2))]

    factors = _factorize2(L)
    DFT_CONST = - 2j * np.pi / L
    if len(factors) == 2:
        # Cooley-Tukey for composite L = L2 * L1, L2 is prime.
        L2, L1 = factors
        (perm1, rot1), (perm2, rot2) = [
                _fermionic_ft_plan(factor) for factor in (L1, L2)]
        assert perm2 is None # Because L2 is prime
        rot2 = _repeat_rotations(rot2, L1)
        twiddle = np.exp(DFT_CONST * np.arange(L2)[:, None]
                         * np.arange(L1)[None, :]).ravel()
        rot2 = _push_diag_to_rotations(twiddle, rot2)
        rot1 = _tile_rotations(rot1, L1, L2)

        # Dit-reversal:
        idx = np.arange(L)
        j1 = idx // L2
        j2 = idx % L2
        idx1 = j2 * L1 + j1
        if perm1 is None:
            perm = idx1
        else:
            perm1 = _tile_permutation(perm1, L2)
            perm = perm1[idx1]

        return perm, rot1 + rot2
    else:
        assert len(factors) == 1
        # L is prime
        # Implement odd-prime case using givens_decompose
        f = dft_matrix(L)
        # 1. Apply Hadamards to make the DFT matrix block-diagonal
        block_end = (L+1)//2
        hadamards = [(h, (i, L-i)) for i in range(1, block_end)]
        hadamard_prod = classical_reconstruct_f(L, None, hadamards)
        f = hadamard_prod @ f @ hadamard_prod
        assert np.allclose(f @ f.conj().T, np.eye(L))
        assert np.allclose(f[:block_end, block_end:], 0)
        # Decompose each block using givens_decompose
        block_1_rotations = givens_decompose(f[:block_end, :block_end])
        block_2_rotations = givens_decompose(f[block_end:, block_end:])
        block_2_rotations = [
            (g, (i + block_end, j + block_end))
            for g, (i, j) in block_2_rotations]
        return None, (
            hadamards + block_1_rotations + block_2_rotations + hadamards)

def classical_lapply_rotation_inplace(g, rows, a):
    """
    Multiply a matrix `a` by a single matrix on the left at positions `rows`.
    """
    rows = list(rows)
    a[rows, :] = g @ a[rows, :]

def classical_reconstruct_f(L, perm, rotations):
    """
    Reconstruct the matrix F from the plan.
    """
    f = np.eye(L, dtype=np.complex128)
    for g, ij in reversed(rotations):
        classical_lapply_rotation_inplace(g, ij, f)
    if perm is not None:
        f = f[perm, :]
    return f

_zz = np.array([[1, -1], [-1, 1]], dtype=np.double)

def _apply_rotation_inplace(rotation, psi):
    """
    Apply a rotation of creation operators to the vector $\ket{\psi}$.

    Usage:
        # Use the same variable psi to store the result:
        psi = _apply_rotation(rotation, psi)

    Args:
        rotation: A tuple (g, (i0, i1)), where g is a 2x2 unitary matrix and
            i0, i1 are the indices of the rows on which g acts
            (0 <= i0 < i1 < L, where L is the number of qubits).
        psi: The vector $\ket{\psi}$ with $2^L$ amplitudes.
            Note: this function modifies psi in-place.

    Returns:
        psi - the updated vector, which may be a view of the original psi.
    """
    # Let v = (z[i0], z[i1]) be the two bits at positions i0, i1.
    # There are 3 cases:
    # 1. v = (0, 0). The rotation acts trivially.
    # 2. v = (1, 0) and v = (0, 1). The rotation acts as
    #   multiplication by Z^s g Z^s, where s is the parity of the number of
    #   1s in z[i0+1:i1] (if we interpret z as a bitstring).
    # 3. v = (1, 1). The rotation acts as multiplication by det(g).
    g, (i0, i1) = rotation
    assert i0 < i1
    det_g = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    L = int(np.log2(len(psi)) + 0.5)

    psi = psi.reshape((1 << (L - i1 - 1), 2, 1 << (i1 - i0 - 1), 2, 1 << i0))
    psi = psi.transpose((0, 2, 4, 1, 3)) # (high, mid, low, i1, i0)
    psi[:, :, :, 1, 1] *= det_g
    has_mid = i1 - i0 > 1
    if has_mid:
        zs_mid = np.arange(1 << (i1 - i0 - 1), dtype=np.uint32)
        assert psi.shape[1] == len(zs_mid)
        zs_mid_sign = 1 - ((bits.countbits32(zs_mid) & 1) << 1).astype(np.int32)
        psi[:, :, :, 1, 0] *= zs_mid_sign[None, :, None]
    # Consider vector (a0, a1)^T = (psi[:, :, :, 0, 1], psi[:, :, :, 1, 0])^T
    # Fix the first 3 coordinates. This vector corresponds to the state
    # a = a0 * c^\dagger_{i0} + a1 * c^\dagger_{i1} = a^T @ c^\dagger
    # We want to replace c^\dagger with g @ c^dagger, i.e. this
    # vector gets replaced with a^T @ (g @ c^\dagger) = (a^T @ g) @ c^\dagger.
    # In other words, we need to perform the transform a := a @ g
    psi[:, :, :, [0, 1], [1, 0]] = psi[:, :, :, [0, 1], [1, 0]] @ g
    if has_mid:
        psi[:, :, :, 1, 0] *= zs_mid_sign[None, :, None]
    psi = psi.transpose((0, 3, 1, 4, 2)) # (high, i1, mid, i0, low)
    psi = psi.reshape(-1)
    return psi

def apply_permutation_inplace(perm, psi):
    """
    Apply a permutation of creation operators to the vector $\ket{\psi}$.

    Usage:
        # Option 1: use the same variable psi to store the result:
        psi = apply_permutation_inplace(perm, psi)
        # Option 2: apply .copy() to psi when calling:
        res = apply_permutation_inplace(perm, psi.copy())

    Note:
        res[z] = (-1)^{parity(z, perm)} * psi[z1],
        where z1 = \sum_j (1<<perm^{-1}[j]) * z[j]
        z1[j] = z[perm[j]]
        parity(z, perm) = \sum_{i<j} z[i] * z[j] * (perm[i] > perm[j])

    Args:
        perm: A permutation of the creation operators.
        psi: The vector $\ket{\psi}$ with $2^L$ amplitudes.

    Returns:
        psi - the updated vector, which may be a view of the original psi.
    """
    if perm is None:
        return psi
    perm = np.asarray(perm, dtype=np.uint32)
    L = len(perm)
    assert len(psi) == (1 << L)

    # Step 1: compute parity in O(2^L) time.
    # parity(z, perm) = \sum_{i<j} z[i] * z[j] * (perm[i] > perm[j])
    parity = np.zeros(1 << L, dtype=np.int32)
    zs = np.arange(1 << L, dtype=np.int32)
    for j in range(1, L):
        # mask = sum(1 << i for i s.t. perm[i] > perm[j])
        ii = np.arange(j)
        mask = np.sum((1 << ii) * (perm[ii] > perm[j]))
        parity[1<<j:1<<(j+1)] = (
            parity[0:1<<j] ^ bits.countbits32(mask & zs[:1<<j]))
    # Convert parity to sign: sign = (-1)^parity
    parity = 1 - ((parity & 1) << 1).astype(np.int32)

    # Step 2: compute z1s (consuming zs)
    perm_inv = np.zeros(L, dtype=np.uint32)
    perm_inv[perm] = np.arange(L, dtype=np.uint32)
    # axes order is opposite to bit order, so we need to relabel perm_inv.
    z1s = zs.reshape((2,) * L).transpose(L-1-perm_inv[::-1]).reshape(-1)
    del zs # Ensure we do not use zs below, as it now contains garbage.

    # Step 3: apply to psi
    psi *= parity
    psi[:] = psi[z1s]
    return psi

def fermionic_fourier(psi, inverse=False, copy=True):
    """
    Apply the fermionic Fourier transform to the vector $\ket{\psi}$.

    Args:
        psi: The vector $\ket{\psi}$ with $2^L$ amplitudes.
        inverse: If True, apply the inverse Fourier transform.
        copy: If True, the input vector is not modified.

    Returns:
        The vector $\ket{\phi}$ containing the amplitudes of the Fourier
        transform.
    """
    if copy:
        psi = psi.copy()
    if len(psi) <= 2:
        return psi
    L = int(np.log2(len(psi)) + 0.5)
    assert 2 <= L
    assert L <= 31

    perm, rotations = _fermionic_ft_plan(L)
    if inverse:
        for g, ij in reversed(rotations):
            g_inv = g.conj().T
            psi = _apply_rotation_inplace((g_inv, ij), psi)
        if perm is not None:
            perm_inv = np.zeros(L, dtype=np.uint32)
            perm_inv[perm] = np.arange(L, dtype=np.uint32)
            psi = apply_permutation_inplace(perm_inv, psi)
        return psi

    if perm is not None:
        psi = apply_permutation_inplace(perm, psi)
    for rotation in rotations:
        psi = _apply_rotation_inplace(rotation, psi)
    return psi
