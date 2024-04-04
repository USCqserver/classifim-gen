import unittest
import numpy as np
import classifim_gen.evec_continuation as evec_continuation
import classifim_gen.hamiltonian as hamiltonian
import classifim_gen.linalg as linalg


class TestEvecContinuation(unittest.TestCase):
    def test_evec_continuation_full_space(self):
        space_dim = 10
        h_int = np.diag(np.arange(space_dim, dtype=np.float64) - (space_dim - 1) / 2)
        h_jump = -np.eye(space_dim, dtype=np.float64)
        h_jump = np.hstack([h_jump[:, 1:], h_jump[:, [0]]])
        h_jump = h_jump + h_jump.T
        h_family = hamiltonian.DenseHamiltonianFamily(
            h_term_matrices=[h_int, h_jump])
        vecs0 = [np.pad(v, (0, space_dim - len(v))) for v in [[1], [0.5, 0.5]]]
        vecs0.append(np.full(shape=space_dim, fill_value=space_dim**(-0.5)))
        vecs0 = np.array(vecs0)
        evc = evec_continuation.EvecContinuator(vecs0, h_family)

        self.assertEqual(vecs0.shape, (3, 10))
        # The vectors of vecs0 are linearly independent, so the subspace
        # spanned by them is 3-dimensional:
        self.assertEqual(evc.ev_continuation_vecs.shape, vecs0.shape)
        self.assertEqual(evc.h_term_matrices.shape, (2, 3, 3))

        params = (1, 1)
        k = 2
        h_total = h_int + h_jump
        np.testing.assert_array_almost_equal_nulp(h_total, h_total.T)
        expected_evals, expected_evecs = np.linalg.eigh(h_total)
        expected_evals = expected_evals[:k]
        expected_evecs = expected_evecs.T[:k, :]
        res = evc.compute_eigenstates(
            params, k, k_init=3, n_iter=7, hard_cutoff=0)
        try:
            np.testing.assert_allclose(
                res["vals"], expected_evals, rtol=1e-14, atol=1e-14)
        except AssertionError:
            print(f"{res['vals']=}")
            print(f"{expected_evals=}")
            err = evc._orthonormality_error(res["vecs"])
            print(f"res['vecs'] orthonormality error: {err}")
            linalg.print_float_matrix(res["vecs"], "res['vecs']")
            linalg.print_float_matrix(expected_evecs, "expected_evecs")
            raise
        sign = 2 * (np.sum(res["vecs"] * expected_evecs, axis=1) > 0) - 1
        assert np.all(np.abs(sign) == 1)
        assert sign.shape == (k,)
        np.testing.assert_allclose(
            res["vecs"], expected_evecs * sign[:, np.newaxis],
            rtol=1e-10, atol=1e-10)


class TestEvecContinuatorBuilder(unittest.TestCase):
    def test_evec_continuator_builder(self):
        ham = hamiltonian.DenseHamiltonianFamily([np.eye(10)])
        builder = evec_continuation.EvecContinuatorBuilder(max_rows=4, hamiltonian_family=ham)
        vecs0 = [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [0, 0, 1],
            [3, 2, 1]]
        vecs0 = np.array(vecs0) / np.linalg.norm(vecs0, axis=1)[:, np.newaxis]
        vecs0 = np.pad(vecs0, ((0, 0), (1, 6)))
        self.assertEqual(vecs0.shape, (6, 10))
        for v in vecs0:
            builder.add_vec(v)
        evc1 = builder.build()
        evc2 = evec_continuation.EvecContinuator(vecs0, hamiltonian_family=ham)
        self.assertEqual(evc1.ev_continuation_vecs.shape, evc2.ev_continuation_vecs.shape)
        signs = np.sign(np.sum(evc1.ev_continuation_vecs * evc2.ev_continuation_vecs, axis=1))
        expected_evc_vecs = signs[:, np.newaxis] * evc2.ev_continuation_vecs
        try:
            np.testing.assert_allclose(
                evc1.ev_continuation_vecs,
                expected_evc_vecs,
                rtol=1e-14, atol=1e-13)
        except AssertionError:
            print(f"{evc1.ev_continuation_vecs.shape=}")
            linalg.print_float_matrix(
                evc1.ev_continuation_vecs, "evc1.ev_continuation_vecs")
            linalg.print_float_matrix(expected_evc_vecs, "expected_evc_vecs")
            raise

        np.testing.assert_allclose(
            evc1.h_term_matrices, evc2.h_term_matrices,
            rtol=1e-14, atol=1e-14)
