import unittest
import classifim_bench.fil24_hamiltonian as fil24_hamiltonian

class TestBraceletCount(unittest.TestCase):
    def test_l12(self):
        diff_cur, total_cur, diff_total, total_total = (
            fil24_hamiltonian.bracelet_count(
                length=12, symmetry_type="", num_bead_types=4))
        # Expected value is computed in by WolframAlpha query
        # "Total number of different necklaces of length 12 with 4 bead types"
        self.assertEqual(diff_total, 704370)
