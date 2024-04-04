import unittest
import numpy as np
import math

import classifim_bench.fft as fft

class TestFFT(unittest.TestCase):
    def test_apply_creation(self):
        psi = np.array([1, 2, 3, 4], dtype=np.complex64)
        res = fft.apply_creation(0, psi)
        res_expected = np.array([0, 1, 0, -3], dtype=np.complex64)
        self.assertTrue(np.allclose(res, res_expected))

    def test_apply_annihilation(self):
        psi = np.array([1, 2, 3, 4], dtype=np.complex64)
        res = fft.apply_annihilation(0, psi)
        res_expected = np.array([2, 0, -4, 0], dtype=np.complex64)
        self.assertTrue(np.allclose(res, res_expected))

    def check_commutation(self):
        psi = np.arange(1 << 4, dtype=np.complex64) + 1
        # Creation operators should anticommute:
        c1c2_psi = fft.apply_creation(1, fft.apply_creation(2, psi))
        c2c1_psi = fft.apply_creation(2, fft.apply_creation(1, psi))
        self.assertTrue(np.allclose(c1c2_psi, -c2c1_psi))

    def test_fermionic_ft_plan(self):
        expected_sizes = [
                0, 0, 1, 3, 4, 8, 9, 15, 12, 18, 21, 35, 24, 48, 37,
                39, 32, 80, 45, 99, 52]
        for L in range(1, 21):
            perm, rotations = fft._fermionic_ft_plan(L)
            res_f = fft.classical_reconstruct_f(L, perm, rotations)
            self.assertTrue(
                np.allclose(res_f, fft.dft_matrix(L)),
                f"fermionic_ft_plan for {L=} is wrong.")
            actual_size = len(rotations)
            expected_size = expected_sizes[L]
            self.assertLessEqual(actual_size, expected_size)

    def slow_fft(self, psi, inverse=False):
        """
        Slow implementation of the FermionicFT.

        This is used to test the fast implementation.

        complexity: O(2^L + nnz * L^2 * 2^L), where nnz is the number of
        non-zero elements in psi.
        """
        L = int(np.log2(len(psi)) + 0.5)
        assert len(psi) == 1 << L
        res = np.zeros_like(psi)
        ft_factor0 = (1 if inverse else -1) * 2j * np.pi / L
        for z in range(1 << L):
            psi_z = psi[z]
            if psi_z == 0:
                continue
            cur_res = np.zeros_like(psi)
            cur_res[0] = psi_z
            for i in range(L):
                if not ((z >> i) & 1):
                    continue
                ft_factor = ft_factor0 * i
                creation_ft = L**(-0.5) * np.exp([
                    ft_factor * j for j in range(L)])
                new_cur_res = np.zeros_like(psi)
                for j in range(L):
                    new_cur_res += (
                        fft.apply_creation(j, cur_res) * creation_ft[j])
                cur_res = new_cur_res
            res += cur_res
        return res

    def test_apply_permutation_inplace(self):
        L = 10
        perm = [0, 5, 1, 6, 2, 9, 3, 8, 7, 4]
        psi = np.zeros(1 << L, dtype=np.complex128)
        psi[0x0f] = 1.0 + 3.0j

        # 0x0f has bits 3, 2, 1, 0 -> 6, 1, 5, 0 (sign = -1 to flip 1 and 5)
        res_expected = np.zeros_like(psi)
        self.assertEqual(0x63, sum(1 << perm[j] for j in range(4)))
        res_expected[0x63] = -1.0 - 3.0j

        res_actual = fft.apply_permutation_inplace(perm, psi.copy())
        self.assertTrue(
            np.allclose(res_actual, res_expected),
            f"{res_actual[0x63]}, {res_expected[0x63]}")


    def _print_vector(self, a, end="\n"):
        print("[", end="")
        for j in range(a.shape[0]):
            if j > 0:
                print(", ", end="")
            a_str = f"{a[j]:.3f}"
            print(f"{a_str:>13s}", end="")
        print("]", end=end)

    def _print_matrix(self, a):
        for i in range(a.shape[0]):
            if i == 0:
                print("[", end="")
            else:
                print(" ", end="")
            self._print_vector(a[i, :], end="")
            if i == a.shape[0] - 1:
                print("]")
            else:
                print("")

    def _test_fermionic_fourier_dense(self, L):
        psi = np.arange(1 << L, dtype=np.complex128) + (math.pi - 3) * 1j
        psi[0x55555555 & ((1 << L) - 1)] = -2**0.5 * 1j
        expected_res = self.slow_fft(psi)
        res = fft.fermionic_fourier(psi)
        self.assertEqual(res.dtype, np.complex128)
        self.assertEqual(res.shape, (1 << L,))
        self.assertTrue(np.allclose(res, expected_res))

    def _test_fermionic_fourier_sparse(self, L, inverse=False):
        # Ensure we only set a couple of elements so
        # slow_fft completes in reasonable time.
        psi = np.zeros(1 << L, dtype=np.complex128)
        l_mask = ((1 << L) - 1)
        psi[0x55555555 & l_mask] = -5**0.5 * 1j
        psi[0xf & l_mask] = -2**0.5 * 1j
        psi[l_mask] = 3**0.5

        expected_res = self.slow_fft(psi, inverse=inverse)
        res = fft.fermionic_fourier(psi, inverse=inverse)
        self.assertEqual(res.dtype, np.complex128)
        self.assertEqual(res.shape, (1 << L,))
        self.assertTrue(np.allclose(res, expected_res))

    def test_fermionic_fourier_3(self):
        self._test_fermionic_fourier_dense(3)

    def test_fermionic_fourier_4(self):
        self._test_fermionic_fourier_dense(4)

    def test_fermionic_fourier_10(self):
        self._test_fermionic_fourier_sparse(10)

    def test_fermionic_fourier_inv(self):
        self._test_fermionic_fourier_sparse(10, inverse=True)
