"""
Basic Hamiltonians implemented in QuSpin.
"""

class KitaevChainFamily:
    """
    Kitaev chain family of Hamiltonians.
    """
    def __init__(self, L, t, delta, mu, bc, dtype=np.float64):
        """
        Initialize a Kitaev chain family of Hamiltonians.

