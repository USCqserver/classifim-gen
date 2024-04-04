import unittest
import numpy as np
import classifim_gen.linalg as linalg
import math


class TestGramSchmidt(unittest.TestCase):
    @staticmethod
    def _test_gram_schmidt(version):
        vecs = np.array(
            [[0.9, 0, 0, 0], [0.1, 1.1, 0, 0], [-0.15, -0.2, 0.95, 0]],
            dtype=np.float64)
        expected_basis = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        expected_transform = vecs[:, :3]
        if version == 1:
            basis = linalg.gram_schmidt(vecs)
        else:
            basis, transform = linalg.gram_schmidt2(vecs)
            np.testing.assert_allclose(transform, expected_transform, rtol=2**(-52), atol=2**(-52))
        np.testing.assert_allclose(basis, expected_basis, rtol=2**(-52), atol=2**(-52))

    def test_gram_schmidt(self):
        self._test_gram_schmidt(version=1)

    def test_gram_schmidt2(self):
        self._test_gram_schmidt(version=2)


class TestFullReorthLanczos(unittest.TestCase):
    @staticmethod
    def slow_reorth_lanczos(apply_h, basis0, n_iter):
        """
        Simpler, but slower and more numerically unstable than linalg.full_reorth_lanczos.
        """
        basis = [b.astype(np.longdouble) for b in basis0]
        for i in range(n_iter):
            basis.append(linalg.make_orthonormal(basis, apply_h(basis[-1])))
        basis = linalg.gram_schmidt(np.array(basis))
        hbasis = np.array([apply_h(v) for v in basis])
        h_matrix = basis @ hbasis.T
        return basis, h_matrix

    @staticmethod
    def print_matrix(m):
        for mi in m:
            print(", ".join(f"{mij:+#.5f}" for mij in mi))

    def test_full_reorth_lanczos(self):
        space_dim = 10
        n_iter = 3
        h = np.diag(np.arange(space_dim, dtype=np.float64))
        h[0, 2] += 2.5
        h[1, 3] += 1.5
        h[0, 1] += -3.25
        h = (h + h.T) / 2
        h_longdouble = h.astype(np.longdouble)
        basis0 = [np.pad(v, (0, space_dim - len(v))) for v in [[1], [0, 1]]]
        v = np.full(shape=space_dim-2, fill_value=(space_dim - 2)**(-0.5))
        v = np.pad(v, (2, 0))
        basis0.append(v)
        expected_basis, expected_h_matrix = self.slow_reorth_lanczos(
                apply_h=lambda v: h_longdouble @ v,
                basis0=basis0,
                n_iter=n_iter)
        basis, h_matrix = linalg.full_reorth_lanczos(
                apply_h=lambda v: h @ v,
                basis0=basis0,
                n_iter=n_iter)
        tol = 5e-15
        try:
            np.testing.assert_allclose(basis, expected_basis,
                                       rtol=tol, atol=tol)
        except AssertionError:
            print("expected_basis:")
            self.print_matrix(expected_basis)
            print("basis:")
            self.print_matrix(basis)
            max_abs_diff = np.max(np.abs(basis - expected_basis))
            num_digits = int(1-math.log(max_abs_diff) / math.log(10))
            print(f"1e{num_digits} * (basis - expected_basis):")
            self.print_matrix(10**num_digits * (basis - expected_basis))
            print("gram matrix:")
            self.print_matrix(basis @ basis.T)
            raise

        np.testing.assert_allclose(h_matrix, expected_h_matrix,
                                   rtol=tol, atol=space_dim * tol)


class TestIsDtypeRealOnly(unittest.TestCase):
    def test_is_dtype_real_only(self):
        self.assertTrue(linalg.is_dtype_real_only(np.float16))
        self.assertTrue(linalg.is_dtype_real_only(np.float64))
        self.assertTrue(linalg.is_dtype_real_only(np.int64))
        self.assertFalse(linalg.is_dtype_real_only(np.csingle))
        self.assertFalse(linalg.is_dtype_real_only(np.complex128))
        dtype_bool = np.array([True]).dtype
        self.assertFalse(linalg.is_dtype_real_only(dtype_bool))
        self.assertFalse(linalg.is_dtype_real_only(np.bool_))
        dtype_string = np.array(["a"]).dtype
        self.assertFalse(linalg.is_dtype_real_only(dtype_string))
        dtype_bytes = np.array([b"a"]).dtype
        self.assertFalse(linalg.is_dtype_real_only(dtype_bytes))
