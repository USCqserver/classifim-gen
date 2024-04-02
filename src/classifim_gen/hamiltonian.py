import abc
import numpy as np
from typing import List, Callable, Union

class LinearHamiltonianFamily(abc.ABC):
    """
    A family of Hamiltonians of the form $H = sum_i params_i H_i$.
    """

    @property
    @abc.abstractmethod
    def h_terms(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        """
        Return a list of functions representing the terms of the Hamiltonian.

        The functions should take a vector and return a matrix representing
        the action of the corresponding term of the Hamiltonian on the vector.
        """
        pass

    def apply_h(self, params_vec: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """
        Apply the Hamiltonian to a vector.

        :param params_vec: Vector of the parameters of the Hamiltonian.
        :param vec: Vector to apply the Hamiltonian to.
        :return: Result of applying the Hamiltonian to the vector.

        The default implementation uses `h_terms` property but can
        be redefined for efficiency reasons.
        """
        params_vec = np.array(params_vec)
        h_terms = self.h_terms
        assert params_vec.shape == (len(h_terms),), f"{params_vec.shape=} != {len(h_terms)=}"
        return np.tensordot(
            params_vec, [h_term(vec) for h_term in h_terms], axes=1)

    @property
    @abc.abstractmethod
    def space_dim(self) -> int:
        """
        Return the dimension of the Hilbert space of the Hamiltonian.
        """
        pass


class DenseHamiltonianFamily(LinearHamiltonianFamily):
    """
    A family of Hamiltonians of the form $H = sum_i params_i H_i$.

    The terms $H_i$ are dense matrices.
    """

    def __init__(self, h_term_matrices: Union[np.ndarray, List[np.ndarray]]):
        """
        :param h_term_matrices: Array with shape (num_h_terms, space_dim, space_dim)
            containing the matrices representing the terms of the Hamiltonian.
        """
        self.h_term_matrices = h_term_matrices
        self._h_terms = [
            (lambda m: lambda v: m @ v)(h_term_matrix)
            for h_term_matrix in self.h_term_matrices]

    @property
    def h_terms(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return self._h_terms

    @property
    def space_dim(self) -> int:
        return self.h_term_matrices[0].shape[0]