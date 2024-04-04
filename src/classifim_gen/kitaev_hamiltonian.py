"""
Functions and classes relevant to the Hubbard Hamiltonian for a 1D chain.
"""

import numpy as np
import quspin
import classifim_gen.hamiltonian as hamiltonian
import classifim_gen.fft as fft

import quspin.operators
import quspin.basis

class KitaevFamily():
    def __init__(self, L):
        """
        Construct the Kitaev Hamiltonian family.

        Args:
            L: The number of sites.
        """
        self.L = L
        # Basis for QuSpin:
        self.basis_qs = None

    def get_basis_qs(self):
        basis = self.basis_qs
        if basis is None or basis.L != self.L:
            basis = quspin.basis.spinless_fermion_basis_1d(self.L, Nf=None)
            self.basis_qs = basis
        return basis

    def get_quspin_ham(self, t, delta, mu, dtype=np.float64):
        """
        Returns quspin's Hamiltonian for given values of the parameters.
        """
        L = self.L
        static_terms = []

        # Hopping terms: -t (c_i^\dagger c_{i+1} + c_{i+1}^\dagger c_i)
        static_terms.append([
            "+-", [[-t, (j+1) % L, j] for j in range(L)]
                + [[-t, j, (j+1) % L] for j in range(L)]
        ])

        # Pairing terms
        static_terms.append([
            "--", [[-delta, (j+1) % L, j] for j in range(L)]])
        static_terms.append([
            "++", [[-delta, j, (j+1) % L] for j in range(L)]])

        # Chemical potential terms
        static_terms.append(["z", [[-mu, j] for j in range(L)]])
        dynamic_terms = []

        # Create Hamiltonian
        ham = quspin.operators.hamiltonian(
            static_terms, dynamic_terms, dtype=dtype, basis=self.get_basis_qs())
        return ham

    def _get_quspin_momentum_static_terms(
            self, t, delta, mu, dtype=np.complex128, j=None):
        L = self.L
        if j is None:
            res = [self._get_quspin_momentum_static_terms(
                t, delta, mu, dtype, j) for j in range((L + 2) // 2)]
            # sum of lists is concatenation:
            return sum(res, start=[])

        v_type = 0 if (2 * j) % L == 0 else 1
        mu_tilde = mu + 2 * t * np.cos(2 * np.pi * j / self.L)
        if v_type == 0:
            return [["z", [[-mu_tilde, j]]]]
        delta_tilde = 2 * delta * np.sin(2 * np.pi * j / self.L)
        return [
                ["z", [[-mu_tilde,j], [-mu_tilde, L-j]]],
                ["--", [[1j * delta_tilde, L-j, j]]],
                ["++", [[1j * delta_tilde, L-j, j]]]
        ]

    def get_quspin_momentum_ham(
            self, t, delta, mu, dtype=np.complex128, j=None):
        """
        Return Kitaev Chain Hamiltonian in the momentum basis in QuSpin format.

        If `j` is given, only the component affecting momenta j and -j
        is returned.
        """
        return quspin.operators.hamiltonian(
            static_list=self._get_quspin_momentum_static_terms(
                t, delta, mu, dtype, j),
            dynamic_list=[],
            dtype=dtype,
            basis=self.get_basis_qs()
        )

    def _get_gs_v(self, t, delta, mu, v_type, j):
        """
        Returns vector v_{tj} from the exact ground state solution.

        Args:
            t, delta, mu: parameters of the Hamiltonian.
            v_type: 0 or 1, type of the vector
            j: momentum index

        Returns:
            v_{tj}: np.ndarray with shape=(2,) and dtype=np.complex128
        """
        mu_tilde = mu + 2 * t * np.cos(2 * np.pi * j / self.L)
        if v_type == 0:
            if mu_tilde < 0:
                return np.array([1.0, 0.0], dtype=np.complex128)
            else:
                return np.array([0.0, 1.0], dtype=np.complex128)
        assert v_type == 1
        delta_tilde = 2 * delta * np.sin(2 * np.pi * j / self.L)
        # We take abs(mu_tilde) to avoid catastrophic cancellation:
        res = np.array([1j * delta_tilde,
            np.abs(mu_tilde) + np.sqrt(mu_tilde**2 + delta_tilde**2)],
            dtype=np.complex128)
        if mu_tilde < 0:
            # The above solution is for abs(mu_tilde) instead of mu_tilde.
            # X @ \tilde{M}_{1j} @ X flips the signs of mu_tilde and delta_tilde
            res[0] *= -1
            res = res[::-1]
        return res / np.linalg.norm(res)

    def get_momentum_basis_ground_state(self, t, delta, mu):
        """
        Returns the ground state of the Hamiltonian in the momentum basis.
        """
        psi = np.zeros(1 << self.L, dtype=np.complex128)
        new_psi = np.zeros(1 << self.L, dtype=np.complex128)
        psi[0] = 1.0
        psi_len = 1
        bit_perm = []
        for j in range((self.L + 2) // 2):
            bit_perm.append(j)
            if (2 * j) % self.L == 0:
                v_type = 0
                new_len = psi_len * 2
            else:
                v_type = 1
                new_len = psi_len * 4
                bit_perm.append(self.L - j)
            v = self._get_gs_v(t, delta, mu, v_type, j)
            new_psi[:psi_len] = psi[:psi_len] * v[0]
            new_psi[new_len - psi_len:new_len] = psi[:psi_len] * v[1]
            psi_len = new_len
            psi, new_psi = new_psi, psi
        assert psi_len == 1 << self.L, f"{psi_len} != {1 << self.L}"
        assert len(bit_perm) == self.L, f"{len(bit_perm)} != {self.L}"
        # Bits are [0, 1, L-1, 2, L-2, ...]
        # We need to reorder them to [0, 1, 2, ..., L-1]
        psi = fft.apply_permutation_inplace(np.array(bit_perm), psi)
        return psi

    def get_coordinate_basis_ground_state(self, t, delta, mu):
        """
        Returns the ground state of the Hamiltonian in the coordinate basis.
        """
        psi = self.get_momentum_basis_ground_state(t, delta, mu)
        psi = fft.fermionic_fourier(psi, inverse=True, copy=False)
        return psi

    def get_param_names(self):
        return ("t", "mu", "delta")
