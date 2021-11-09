"""Tests the 2D numerics library."""
import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import unittest

import numpy as np
import scipy.io as sp_io

from two_d import numerics_2d


class TestNumerics2D(unittest.TestCase):
    """Tests for the 2D numerics library.

    Resutls in this tests are compared with the counterpart in the Matlab
    library.
    """
    def setUp(self):
        """Initializes common variables for the test."""
        super().setUp()

        self.ref = sp_io.loadmat('./test_data/numerics_2d_ref.mat')

    def test_numerical_operators(self):
        """Checks results from numerics_2d."""
        n = 6

        x, y = numerics_2d.nodes_2d(n)
        with self.subTest('Node2D'):
            np.testing.assert_array_almost_equal( \
                    np.squeeze(self.ref['x']), x, decimal=2)
            np.testing.assert_array_almost_equal( \
                    np.squeeze(self.ref['y']), y, decimal=2)

        r, s = numerics_2d.xy_to_rs(x, y)
        r_ref = np.squeeze(self.ref['r'])
        s_ref = np.squeeze(self.ref['s'])
        with self.subTest('XYToRS'):
            np.testing.assert_array_almost_equal(r_ref, r, decimal=2)
            np.testing.assert_array_almost_equal(s_ref, s, decimal=2)

        a, b = numerics_2d.rs_to_ab(r, s)
        with self.subTest('RSToAB'):
            np.testing.assert_array_almost_equal( \
                    np.squeeze(self.ref['a']), a, decimal=2)
            np.testing.assert_array_almost_equal( \
                    np.squeeze(self.ref['b']), b, decimal=2)

        dmdr, dmds = numerics_2d.grad_simplex_2d_p(a, b, 0, 1)
        with self.subTest('GradSimplex2DP'):
            np.testing.assert_array_almost_equal( \
                    np.squeeze(self.ref['dmdr']), dmdr)
            np.testing.assert_array_almost_equal( \
                    np.squeeze(self.ref['dmds']), dmds)

        vr, vs = numerics_2d.grad_vandermonde_2d(n, r_ref, s_ref)
        with self.subTest('GradVandermonde2D'):
            np.testing.assert_array_almost_equal(self.ref['vr'], vr, decimal=5)
            np.testing.assert_array_almost_equal(self.ref['vs'], vs, decimal=5)

        v = numerics_2d.vandermonde_2d(n, r_ref, s_ref)
        with self.subTest('Vandermonde2D'):
            np.testing.assert_array_almost_equal(self.ref['v'], v)

        dr, ds = numerics_2d.d_matrices_2d(n, r_ref, s_ref, self.ref['v'])
        with self.subTest('DMatrices2D'):
            np.testing.assert_array_almost_equal(self.ref['dr'], dr)
            np.testing.assert_array_almost_equal(self.ref['ds'], ds)

        inv_v = np.linalg.inv(self.ref['v'])
        f = numerics_2d.filter_2d(n, 1, 2, self.ref['v'], inv_v)
        with self.subTest('Filter2D'):
            np.testing.assert_array_almost_equal(self.ref['f'], f, decimal=2)

        warp = numerics_2d.warp_factor(n, r_ref)
        with self.subTest('WarpFactor'):
            np.testing.assert_array_almost_equal( \
                    np.squeeze(self.ref['warp']), warp)


if __name__ == '__main__':
    unittest.main()
