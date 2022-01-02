"""Tests the library of utility functions for the 2D Euler equation."""
import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import numpy as np
import unittest

from two_d.euler import util


class TestEuler2DUtil(unittest.TestCase):
    """Tests for the 2D Euler utility functions library."""

    def test_fluxes_and_primitive_variables(self):
        """Checks if fluxes and primitive variables are computed correctly."""
        q = np.array([[1.2, 12.0, 24.0, 4e5], [0.5, 10.0, 5.0, 6e5]])
        q = np.expand_dims(q, 0)
        gamma = 1.4

        f, g, rho, u, v, p = util.fluxes_and_primitive_variables(q, gamma)

        with self.subTest(name='F'):
            expected = np.array( \
                [[12.0, 1.6e5, 2.4e2, 5.5988e6], \
                [10.0, 2.4015e5, 100.0, 1.6799e7]])
            expected = np.expand_dims(expected, 0)
            np.testing.assert_array_almost_equal(expected, f)

        with self.subTest(name='G'):
            expected = np.array( \
                [[24.0, 2.4e2, 1.6036e5, 1.11976e7], \
                [5.0, 100.0, 2.4e5, 8.3995e6]])
            expected = np.expand_dims(expected, 0)
            np.testing.assert_array_almost_equal(expected, g)

        with self.subTest(name='rho'):
            expected = np.array([[1.2, 0.5]])
            np.testing.assert_array_almost_equal(expected, rho)

        with self.subTest(name='u'):
            expected = np.array([[10.0, 20.0]])
            np.testing.assert_array_almost_equal(expected, u)

        with self.subTest(name='v'):
            expected = np.array([[20.0, 10.0]])
            np.testing.assert_array_almost_equal(expected, v)

        with self.subTest(name='p'):
            expected = np.array([[1.5988e5, 2.3995e5]])
            np.testing.assert_array_almost_equal(expected, p)

    def test_flux_hll(self):
        """Checks if the HLL flux is computed correctly."""
        n_x = np.array([[0.0, np.sqrt(3.0) / 2.0]])
        n_y = np.array([[-1.0, 0.5]])
        q_m = np.array([[[1.2, 12.0, 24.0, 4e5], [0.5, 10.0, 5.0, 6e5]]])
        q_p = np.array([[[1.6, 16.0, 25.0, 5e5], [0.25, 20.0, 10.0, 8e5]]])
        gamma = 1.4

        flux = util.flux_hll(n_x, n_y, q_m, q_p, gamma)

        expected = np.array( \
                [[[-110.1759, -1.101759e3, -1.815785e5, -3.247229e7], \
                [164.116, 2.314183e5, 1.34e5, -6.614398e7]]])

        # Decimal set to 0 because the absolute values are large. The relative
        # error is O(1e-6).
        np.testing.assert_array_almost_equal(expected, flux, decimal=0)

if __name__ == '__main__':
    unittest.main()
