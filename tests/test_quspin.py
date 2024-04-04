import unittest

import functools
import math
import numpy as np
import quspin
import scipy.sparse

import classifim_gen.fft as fft

# Test QuSpin conventions.
@unittest.skip("""QuSpin index changes over time.
We should fix the main code to not depend on them.""")
class TestFermionQuSpin(unittest.TestCase):
    def _test_ops(self, L, state_int, j):
        fermion_basis = quspin.basis.spinless_fermion_basis_1d(L, Nf=None)
        assert state_int == fermion_basis.state_to_int(f"{state_int:0{L}b}")
        state = np.zeros(2**L, dtype=np.complex128)
        state[state_int] = 1.0
        self._test_op_plus(fermion_basis, state, j)
        self._test_op_minus(fermion_basis, state, j)
        self._test_op_n(fermion_basis, state, j)
        self._test_op_z(fermion_basis, state, j)

    @staticmethod
    def fermion_basis_op(L, basis, opstr, indx):
        me, row, col = basis.Op(opstr=opstr, indx=[indx], J=1.0, dtype=np.complex128)
        assert row.shape == np.unique(row).shape
        assert col.shape == np.unique(col).shape
        return scipy.sparse.coo_matrix((me, (row, col)), shape=(1<<L, 1<<L))

    @staticmethod
    def state_str(state):
        L = int(math.log2(len(state)) + 0.5)
        res = [
            f"{v} * |{i:0{L}b}>" for i, v in enumerate(state) if abs(v) > 1e-10]
        if len(res) == 0:
            return "0"
        return " + ".join(res)

    def _test_op_plus(self, fermion_basis, state, j):
        L = int(math.log2(len(state)) + 0.5)
        fermion_basis_op = functools.partial(
            self.fermion_basis_op, L, fermion_basis)
        actual = fermion_basis_op("+", j) @ state
        expected = fft.quspin_conversion(
                fft.apply_creation(j, fft.quspin_conversion(state)))
        self.assertTrue(
            np.allclose(actual, expected),
            f"op(+, {j}) @ {self.state_str(state)} = "
            + f"{self.state_str(actual)} != {self.state_str(expected)}")

    def _test_op_minus(self, fermion_basis, state, j):
        L = int(math.log2(len(state)) + 0.5)
        fermion_basis_op = functools.partial(
            self.fermion_basis_op, L, fermion_basis)
        actual = fermion_basis_op("-", j) @ state
        expected = fft.quspin_conversion(
                fft.apply_annihilation(j, fft.quspin_conversion(state)))
        self.assertTrue(
            np.allclose(actual, expected),
            f"op(-, {j}) @ {self.state_str(state)} = "
            + f"{self.state_str(actual)} != {self.state_str(expected)}")

    def _test_op_n(self, fermion_basis, state, j):
        L = int(math.log2(len(state)) + 0.5)
        fermion_basis_op = functools.partial(
            self.fermion_basis_op, L, fermion_basis)
        actual = fermion_basis_op("n", j) @ state
        expected = fft.quspin_conversion(
                fft.apply_occupation_number(j, fft.quspin_conversion(state)))
        self.assertTrue(
            np.allclose(actual, expected),
            f"op(n, {j}) @ {self.state_str(state)} = "
            + f"{self.state_str(actual)} != {self.state_str(expected)}")

    def _test_op_z(self, fermion_basis, state, j):
        L = int(math.log2(len(state)) + 0.5)
        fermion_basis_op = functools.partial(
            self.fermion_basis_op, L, fermion_basis)
        actual = fermion_basis_op("z", j) @ state
        expected = fft.quspin_conversion(
                fft.apply_z(j, fft.quspin_conversion(state)))
        self.assertTrue(
            np.allclose(actual, expected),
            f"op(z, {j}) @ {self.state_str(state)} = "
            + f"{self.state_str(actual)} != {self.state_str(expected)}")

    def test_ops(self):
        _test_ops = self._test_ops
        _test_ops(L=4, state_int=0b0000, j=0)
        _test_ops(L=4, state_int=0b0001, j=3)
        _test_ops(L=4, state_int=0b0010, j=0)
        _test_ops(L=4, state_int=0b0100, j=3)
        _test_ops(L=4, state_int=0b1000, j=1)
        _test_ops(L=4, state_int=0b1100, j=2)
        _test_ops(L=4, state_int=0b1010, j=0)
        _test_ops(L=4, state_int=0b0101, j=3)
        _test_ops(L=4, state_int=0b1011, j=1)
        _test_ops(L=4, state_int=0b1111, j=1)

    def test_index(self):
        L = 5
        fermion_basis = quspin.basis.spinless_fermion_basis_1d(L, Nf=None)
        for i in range(1<<L):
            # If this fails, we need to fix fft.quspin_conversion
            self.assertEqual(L - 1 - i, fermion_basis.index(i))
