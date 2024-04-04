import numpy as np
import classifim_gen.linalg as linalg
import gc
from typing import Tuple


class EvecContinuatorBuilder:
    """
    A class for building EvecContinuator objects.

    This class is needed to avoid OOM errors when the number of vectors
    is large (but they still belong to a subspace of small dimension).
    It works by identifying the subspace spanned by a set of vectors
    once the set reaches specified size.

    Otherwise, you could just construct an EvecContinuator object directly.
    """
    def __init__(self, max_rows=768, gc_collect=True, **kwargs):
        self.max_rows = max_rows
        self.gc_collect = gc_collect
        self.res = EvecContinuator.__new__(EvecContinuator)
        self.res.init_params(**kwargs)
        self.verbosity = self.res.verbosity
        self.vecs0 = []
        self.num_vecs0 = 0

    def add_vec(self, vec):
        self.vecs0.append(vec)
        self.num_vecs0 += 1 if len(vec.shape) == 1 else vec.shape[0]
        del vec
        if self.num_vecs0 >= self.max_rows:
            self.compactify_vecs0()

    def compactify_vecs0(self):
        """Change vectors of vecs0 to keep the Gaussian distribution approximately the same.

        The Gaussian distribution is w @ vecs0 where elements of w are i.i.d. standard normal.
        """
        vecs0 = np.vstack(self.vecs0)
        if self.verbosity >= 1:
            print(f"Compactifying: vecs0.shape={vecs0.shape}")
        space_dim = vecs0.shape[1]
        del self.vecs0
        if self.gc_collect:
            gc.collect()
        if self.verbosity >= 1:
            print(f"Computing gram matrix...")
        gram_evals, gram_evecs = self.res._diagonalize_vecs0_gram(vecs0)
        if self.verbosity >= 1:
            print(f"Multiplying {gram_evecs.shape} x {vecs0.shape}...")
        self.vecs0 = [gram_evecs @ vecs0]
        self.num_vecs0 = self.vecs0[0].shape[0]
        assert self.vecs0[0].shape[1] == space_dim
        if self.verbosity >= 1:
            print(f"Result: vecs0.shape={self.vecs0[0].shape}")
        del gram_evals, gram_evecs, vecs0
        if self.gc_collect:
            gc.collect()

    def build(self):
        vecs0 = np.vstack(self.vecs0)
        del self.vecs0
        if self.gc_collect:
            gc.collect()
        self.res._compute_evec_continuator_data(vecs0)
        del vecs0
        if self.gc_collect:
            gc.collect()
        return self.res


class EvecContinuator:
    SAVED_ATTRS = [
        "ev_continuation_vecs", "ev_continuation_vals", "h_term_matrices",
        "cutoff", "abs_cutoff", "rel_cutoff", "verbosity"]
    VERSION = "0.1.0"

    def __init__(self, vecs0: np.ndarray, *args, **kwargs):
        """
        EvecContinuator helps to compute the ground state of the Hamiltonians.

        The Hamiltonian is assumed to come from a linear family.
        Currently only real Hamiltonians and vectors are supported.

        :param vecs0: Array with shape (num_vecs, space_dim) to initialize
                the subspace.
        :param hamiltonian_family: An object of the HamiltonianFamily class.
        :param abs_cutoff: Absolute cutoff for the eigenvalues of the Gram matrix.
        :param rel_cutoff: Coefficient for cutoff for the eigenvalues of the Gram matrix.
            See the comment in _compute_ev_continuation_vecs for details.
        :param verbosity: Verbosity level. 0 means no output.
        """
        self.init_params(*args, **kwargs)
        self._compute_evec_continuator_data(vecs0)

    def init_params(
        self, hamiltonian_family: "LinearHamiltonianFamily",
        abs_cutoff: float = 2 ** (-36), rel_cutoff: float = 3.0, verbosity: int = 0
    ):
        """
        This function performs the cheap part of initialization.

        See __init__'s docstring for more details."""
        self.ham_family = hamiltonian_family
        self.abs_cutoff = abs_cutoff
        self.rel_cutoff = rel_cutoff
        self.verbosity = max(0, verbosity)

    @staticmethod
    def _orthonormality_error(vecs: np.ndarray) -> float:
        """
        Compute the error of the orthonormality of the given vectors.

        The error is defined as the Hilbert-Schmidt norm of the difference
        between the Gram matrix and the identity matrix.
        """
        vecs = np.array(vecs)
        return np.linalg.norm(vecs @ vecs.T - np.eye(vecs.shape[0]))

    def _diagonalize_vecs0_gram(
            self, vecs0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and truncate eigenvalues and eigenvectors of the Gram matrix.
        """
        assert len(vecs0.shape) == 2
        gram_matrix = vecs0 @ vecs0.T
        gram_evals, gram_evecs = np.linalg.eigh(gram_matrix)
        # `eigvals` are guaranteed to be sorted in ascending order.
        self.cutoff = self.abs_cutoff + self.rel_cutoff * max(0, -gram_evals[0])
        cutoff_i = np.searchsorted(gram_evals, self.cutoff)
        num_ev_continuation_vecs = vecs0.shape[0] - cutoff_i
        gram_evals = np.flip(gram_evals[cutoff_i:], axis=0)
        gram_evecs = np.flip(gram_evecs[:, cutoff_i:], axis=1).T
        assert gram_evals.shape == (num_ev_continuation_vecs,), (
            f"{gram_evals.shape=} != ({num_ev_continuation_vecs=},)")
        assert gram_evecs.shape == (num_ev_continuation_vecs, vecs0.shape[0])
        return gram_evals, gram_evecs

    def _compute_ev_continuation_vecs(self, vecs0: np.ndarray, gc_collect=True) -> None:
        """
        Compute the subspace and its basis to be used in eigenvector continuation.

        Computes `self.cutoff` value to serve as the cutoff for Gram matrix
        eigenvalues. The cutoff is computed as
        `abs_cutoff + rel_cutoff * max(0, -minimal_eigenvalue)`.
        Note that exact eigenvalues of the exact Gram matrix are always
        non-negative. Thus, minimal_eigenvalue can only be negative due
        to numerical errors.

        The resulting subspace is represented by `self.ev_continuation_vecs`.
        """
        gram_evals, gram_evecs = self._diagonalize_vecs0_gram(vecs0)
        ev_continuation_vecs0 = (gram_evals[:, np.newaxis] ** (-1 / 2) * gram_evecs) @ vecs0
        if gc_collect:
            del vecs0
            gc.collect()
        self.ev_continuation_vecs = linalg.gram_schmidt(ev_continuation_vecs0)
        self.ev_continuation_vals = gram_evals
        if self.verbosity:
            for i, evecs in enumerate([ev_continuation_vecs0, self.ev_continuation_vecs]):
                err = self._orthonormality_error(evecs)
                print(f"ev_continuation_vecs{i} orthonormality error: {err}")

    def _compute_h_term_matrices(self) -> None:
        """
        Precompute the restriction of the Hamiltonian to the subspace.

        Computes the matrices representing the restriction of the
        terms of the Hamiltonian to the subspace spanned by
        `self.ev_continuation_vecs`.

        The resulting matrices are stored as a 3-dimensional tensor
        `self.h_term_matrices`.
        """
        h_terms = self.ham_family.h_terms
        hevc_terms = np.array([
            [h_term(v) for v in self.ev_continuation_vecs]
            for h_term in h_terms])
        h_term_matrices = hevc_terms @ self.ev_continuation_vecs.T
        self.h_term_matrices = (h_term_matrices + np.transpose(h_term_matrices, axes=(0, 2, 1))) / 2

    def _compute_evec_continuator_data(self, vecs0: np.ndarray) -> None:
        """
        Compute ev_continuation_vecs and h_term_matrices
        """
        vecs0 = np.array(vecs0)
        self._compute_ev_continuation_vecs(vecs0)
        self._compute_h_term_matrices()

    def compute_eigenstates(
            self, params_vec: np.ndarray,
            k: int = 2, k_init: int = 4, n_iter: int = 24,
            verbosity: int = None, **kwargs):
        """
        Compute `k` smallest eigenstates of the Hamiltonian.

        :param params_vec: Vector of the parameters of the Hamiltonian.
        :param k: Number of eigenstates to compute.
        :param k_init: Number of initial eigenstates from the subspace to use.
        :param n_iter: Number of iterations of full orthonormalization Lanczos.
        :param verbosity: Verbosity level. 0 means no output.
        :param kwargs: Additional keyword arguments to be passed to
                       `full_reorth_lanczos`.
        :return: Dictionary with the following keys:
            - `vals`: Array of eigenvalues.
            - `vecs`: Array of eigenvectors.
            - `version`: Version of the algorithm used.
        """
        if verbosity is None:
            verbosity = self.verbosity
        h_matrix = np.tensordot(params_vec, self.h_term_matrices, axes=1)
        _, h_matrix_evecs = np.linalg.eigh(h_matrix)
        h_matrix_evecs = h_matrix_evecs.T[:k_init, :]

        basis = list(h_matrix_evecs[::-1, :] @ self.ev_continuation_vecs)
        if verbosity:
            for name, vecs in [("basis0", basis), ("ev_continuation_vecs", self.ev_continuation_vecs)]:
                err = self._orthonormality_error(vecs)
                print(f"{name} orthonormality error: {err}")

        def apply_h(v):
            return self.ham_family.apply_h(params_vec, v)
        # basis[i] @ ham @ basis[j] = h_matrix[i,j]
        # basis[i] @ basis[j] = delta[i, j]
        basis, h_matrix2 = linalg.full_reorth_lanczos(
            apply_h, basis, n_iter, verbosity=verbosity, **kwargs)
        h_matrix2 = (h_matrix2 + h_matrix2.T) / 2
        h_matrix2_evals, h_matrix2_evecs = np.linalg.eigh(h_matrix2)
        if verbosity:
            for name, vecs in [("basis", basis), ("h_matrix2_evecs", h_matrix2_evecs)]:
                err = self._orthonormality_error(vecs)
                print(f"{name} orthonormality error: {err}")
        return {
            "vals": h_matrix2_evals[:k],
            "vecs": h_matrix2_evecs.T[:k, :] @ basis,
            "version": f"EVC {self.VERSION}"}

    @classmethod
    def from_file(cls, filename: str, ham_family: "LinearHamiltonianFamily"):
        data = np.load(filename)
        obj = cls.__new__(cls)
        obj.ham_family = ham_family
        for attr in cls.SAVED_ATTRS:
            setattr(obj, attr, data[attr])
        return obj

    def save(self, filename: str):
        np.savez_compressed(
            filename,
            version=self.VERSION,
            **{attr: getattr(self, attr) for attr in self.SAVED_ATTRS})
