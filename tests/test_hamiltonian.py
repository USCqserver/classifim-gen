import unittest
import numpy as np
import classifim_gen.hamiltonian as hamiltonian


class TestDenseHamiltonianFamily(unittest.TestCase):
    def setUp(self):
        self.space_dim = space_dim = 10
        self.h_int = np.diag(np.arange(space_dim, dtype=np.float64) - (space_dim - 1) / 2)
        h_jump = -np.eye(space_dim, dtype=np.float64)
        h_jump = np.hstack([h_jump[:, 1:], h_jump[:, [0]]])
        self.h_jump = h_jump + h_jump.T
        self.h_family = hamiltonian.DenseHamiltonianFamily(
            h_term_matrices=[self.h_int, self.h_jump])

    def test_h_terms(self):
        retrieved_hams = [[], []]
        h_terms = self.h_family.h_terms
        for i in range(self.space_dim):
            vi = np.zeros(self.space_dim, dtype=np.float64)
            vi[i] = 1.0
            for j, h_term in enumerate(h_terms):
                retrieved_hams[j].append(h_term(vi))
        retrieved_hams = [np.array(ham) for ham in retrieved_hams]
        np.testing.assert_allclose(
            retrieved_hams[0], self.h_int, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(
            retrieved_hams[1], self.h_jump, rtol=1e-15, atol=1e-15)

    def test_apply_h(self):
        params = (1, 2**0.5)
        expected_h_total = params[0] * self.h_int + params[1] * self.h_jump
        retrieved_h_total = []
        for i in range(self.space_dim):
            vi = np.zeros(self.space_dim, dtype=np.float64)
            vi[i] = 1.0
            retrieved_h_total.append(
                self.h_family.apply_h(params, vi))
        retrieved_h_total = np.array(retrieved_h_total)
        np.testing.assert_allclose(
            retrieved_h_total, expected_h_total, rtol=1e-15, atol=1e-15)
