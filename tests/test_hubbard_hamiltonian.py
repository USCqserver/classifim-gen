import unittest
import numpy as np
import classifim_gen.hubbard_hamiltonian as hubbard_hamiltonian
import os

# cupy is optional:
NUMPY_LIKE_MODULES = [np]
try:
    # noinspection PyUnresolvedReferences
    import cupy
    NUMPY_LIKE_MODULES.append(cupy)
except ImportError:
    print("Cupy not found, skipping tests for it.")
    pass


class TestHubbard1DFamily(unittest.TestCase):
    # GOLDEN_PATH = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), "hubbard12_golden.npz")

    def subtest_hkin_precomp(self, h_kin_precomp):
        """This test verifies the signs in h_kin_precomp."""
        jump_lsts_per_sign = [
            sorted([
                (int(vinew[i]), int(viold[i]))
                for vinew, viold in precomp_s
                for i in range(len(vinew))
            ]) for precomp_s in h_kin_precomp]
        jump_lsts_per_sign_sets = [set(v) for v in jump_lsts_per_sign]
        res = {}
        for i in range(2):
            for vinew, viold in jump_lsts_per_sign[i]:
                present = tuple((viold, vinew) in jump_lsts_per_sign_sets[j] for j in range(2))
                key = (i, present[0], present[1])
                res[key] = res.get(key, 0) + 1
        self.assertEqual(
            sorted(list(res.keys())),
            [(0, True, False), (1, False, True)])

    def test_hkin_precomp(self):
        family = hubbard_hamiltonian.Hubbard1DFamily(dtype_float=np.float64, xp=np)
        for attr in ["h_nn_precomp", "h_nnn_precomp"]:
            with self.subTest(attr=attr):
                h_kin_precomp = getattr(family, attr)
                self.subtest_hkin_precomp(h_kin_precomp)

    # Note: GOLDEN file is not included to reduce the size of the submission.
    # def subtest_hubbard1d_family(self, xp):
    #     family = hubbard_hamiltonian.Hubbard1DFamily(dtype_float=np.float64, xp=xp)
    #     golden = np.load(self.GOLDEN_PATH)
    #     hv = family.apply_h(golden["params"], golden["v0"])
    #     # Check both maximal and average errors:
    #     np.testing.assert_allclose(hv, golden["hv"], rtol=1e-14, atol=1e-14)
    #     rmse = np.linalg.norm(hv - golden["hv"]) / np.linalg.norm(golden["hv"])
    #     self.assertLess(rmse, 1e-14)

    # def test_hubbard1d_family(self):
    #     for xp in NUMPY_LIKE_MODULES:
    #         with self.subTest(xp=xp.__name__):
    #             self.subtest_hubbard1d_family(xp)

    def test_compute_zsp(self):
        nsites = 6
        zs = np.array([0b100000, 0b111001, 0b000000], dtype=np.int16)
        zsp = hubbard_hamiltonian.Hubbard1DFamily.compute_zsp(zs, nsites)
        expected_zsp = np.array([0b111111, 0b101110, 0b000000], dtype=np.int16)
        np.testing.assert_equal(zsp, expected_zsp)

    def test_countbits16(self):
        zs = np.array([0x0000, 0x0001, 0xffff], dtype=np.int16)
        expected = np.array([0, 1, 16], dtype=np.int16)
        assert np.array_equal(hubbard_hamiltonian.countbits16(zs), expected)
