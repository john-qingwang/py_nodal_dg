import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')
import unittest

import numpy as np
import scipy.io as sp_io

import two_d.geometry as geo


class TestGeometry(unittest.TestCase):
    """Checks the correctness of functions in geometry.py."""

    def setUp(self):
        """Initializes common variables in the test."""
        super().setUp()

        self.mesh = sp_io.loadmat('./test_data/maxwell.mat')

    def test_connect_provides_correct_connectivity_matrices(self):
        """Checks if the connect function generates correct 2D connectivity."""
        e_to_e, e_to_f = geo.connect(self.mesh['EToV'])

        with self.subTest(name='EToE'):
            np.testing.assert_array_equal(self.mesh['EToE'], e_to_e)

        with self.subTest(name='EToF'):
            np.testing.assert_array_equal(self.mesh['EToF'], e_to_f)


if __name__ == '__main__':
    unittest.main()
