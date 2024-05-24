import numpy as np
from typing import List, Union


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Orthonormalize an array of vectors.

    :param vectors: Array of vectors to be orthonormalized of shape (num_vecs, space_dim)
                    (i.e. the vectors are in the rows).

    :returns: basis - the orthonormalized vectors of shape (num_vecs, space_dim)
    """
    num_vecs, space_dim = vectors.shape
    assert num_vecs <= space_dim
    if num_vecs == 0:
        return np.zeros(shape=vectors.shape)
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b) * b for b in basis)
        w /= np.linalg.norm(w)
        basis.append(w)
    return np.array(basis)


def make_orthonormal(basis: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Make vector `w` orthonormal to the array of vectors `basis`.
    """
    w = w - sum(np.dot(w, b) * b for b in basis)
    w /= np.linalg.norm(w)
    return w


def gram_schmidt2(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthonormalize an array of vectors.

    :param vectors: Array of vectors to be orthonormalized of shape (num_vecs, space_dim)
                    (i.e. the vectors are in the rows).

    :returns: A pair (basis, transform) where
        - basis is the orthonormalized vectors of shape (num_vecs, space_dim)
        - transform is s.t. vectors = transform @ basis,
        transform.shape == (num_vecs, num_vecs)
    """
    num_vecs, space_dim = vectors.shape
    assert num_vecs <= space_dim
    if num_vecs == 0:
        return np.zeros(shape=vectors.shape)
    basis = []
    transform = []
    for v in vectors:
        proj = [np.dot(v, b) for b in basis]
        w = v - sum(p * b for p, b in zip(proj, basis))
        w_norm = np.linalg.norm(w)
        w /= w_norm
        basis.append(w)
        proj.append(w_norm)
        transform.append(np.pad(proj, pad_width=(0, num_vecs - len(proj))))
    return np.array(basis), np.array(transform)


def full_reorth_lanczos(
        apply_h: np.ndarray, basis0: list[np.ndarray], n_iter: int,
        soft_cutoff: float = 1e-1, hard_cutoff: float = 1e-12,
        verbosity: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively apply Hamiltonian to the last basis element and re-orthonormalize the result.

    That is, this function effectively repeats the following procedure n_iter times:
    1. Take the last basis element v from basis.
    2. Apply H to get H @ v.
    3. Apply Gram-Schmidt iteration to get a vector w.
    4. Add vector w to basis. Due to step 3 basis is kept orthonormal.
    While doing that, we keep track of (basis[i] @ H @ basis[j]) to avoid
    re-computing them (`space_dim` is assumed to be large).

    Currently only complex numbers are not supported, so the Hamiltonian is
    required to be real-symmetric.

    :param apply_h: Hermitian matrix H supplied as a function to apply the matrix to a vector.
                    I.e. for a vector v the vector H @ v should be computed by `apply_h(v)`.
    :param basis0: Initial basis - non-empty list of orthonormal vectors of the same length `space_dim`.
    :param n_iter: Number of iterations to perform.
    :param soft_cutoff: If pre-normalization norm of the vector
                        to be added to the basis <= soft_cutoff,
                        one more orthonormalization attempt is made.
    :param hard_cutoff: If pre-normalization norm of the vector
                        to be added to the basis <= hard_cutoff,
                        the iteration is stopped.
    :param verbosity: Verbosity level (0 means silent).

    :returns: A pair `(basis, h_matrix)` s.t.
        * basis is np.ndarray of shape `(len(basis0) + n_iter, space_dim)`
        * h_matrix is np.ndarray of shape `(len(basis0) + n_iter, len(basis0) + n_iter)`
          s.t. `h_matrix[i, j] = basis[i] @ H @ basis[j] = h_matrix[j, i]`.
    """
    len_basis0 = len(basis0)
    assert len_basis0 >= 1
    assert hard_cutoff <= soft_cutoff
    basis = basis0.copy()
    space_dim, = basis0[0].shape
    assert len_basis0 + n_iter <= space_dim, (
        f"space_dim={space_dim}, len(basis0)={len_basis0}, n_iter={n_iter}")
    # Final length of the basis (after all iterations are complete)
    h_matrix: List[Union[None, List[float], np.ndarray]] = [None] * (len_basis0 - 1)
    for i in range(len(basis0) - 1, len_basis0 + n_iter - 1):
        v = basis[-1]  # == basis[i]
        w = apply_h(v)
        proj = [np.dot(b, w) for b in basis]
        w -= sum(p * b for p, b in zip(proj, basis))
        w_norm = np.linalg.norm(w)
        if w_norm <= hard_cutoff:
            h_matrix.append(proj)
            if verbosity >= 1:
                print(f"full_reorth_lanczos: stopping iterations at i={i} "
                      f"(len_basis0={len_basis0}, w_norm={w_norm}, "
                      f"hard_cutoff={hard_cutoff})")
            break
        w /= w_norm
        if w_norm <= soft_cutoff:
            w = w - sum(np.dot(w, b) * b for b in basis)
            w_norm2 = np.linalg.norm(w)
            if w_norm <= soft_cutoff:
                h_matrix.append(proj)
                if verbosity >= 1:
                    print(f"full_reorth_lanczos: stopping iterations at i={i} "
                          f"(len_basis0={len_basis0}, w_norm={w_norm}, w_norm2={w_norm2})")
                break
            w /= w_norm2
        basis.append(w)
        proj.append(w_norm)
        h_matrix.append(proj)
    else:
        # Never reached the hard cutoff:
        h_matrix.append(None)

    len_basis = len(basis)
    assert len(h_matrix) == len_basis
    basis = np.array(basis)
    for i in range(len(h_matrix)):
        cur_row = h_matrix[i]
        if cur_row is None:
            hv = apply_h(basis[i])
            h_matrix[i] = basis @ hv
        else:
            h_matrix[i] = np.pad(cur_row, pad_width=(0, len_basis - len(cur_row)))
    return basis, np.column_stack(h_matrix)


def print_float_matrix(m, name=None):
    """Print a float matrix in a nice way."""
    if name is not None:
        print(f"{name}:")
    m = np.array(m)
    assert len(m.shape) == 2
    for mi in m:
        print(", ".join(f"{mij:+#.5f}" for mij in mi))

def is_dtype_real_only(dtype):
    """Check if the given dtype is real only (i.e. not complex).

    More specifically, this should return True iff the dtype can hold
    some real numbers (e.g. 0 or 1) but not other objects (e.g. complex numbers).
    NA is Ok but True is not.

    Note that `np.array([2**64]).dtype` could be `object` and thus this function
    can return False (since, e.g., complex numbers can be stored as objects).

    :param dtype: numpy dtype to check.
    :returns: True if the dtype is real only, False otherwise.
    """
    is_number = np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
    is_complex = np.issubdtype(dtype, np.complexfloating)
    return is_number and not is_complex
