"""Test the Equation library for 2D nodal DG discretization."""
import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import unittest

import numpy as np
import scipy.io as sp_io

from two_d import equation


class TestEquation(unittest.TestCase):
    """A library that tests the Equation utilities.

    Results in this library are validated against the Maxwell simulation
    obtained from the Matlab counterpart.
    """

    def setUp(self):
        """Initializes common variables for the test."""
        super().setUp()

        mesh = sp_io.loadmat('./test_data/maxwell.mat')
        self.ref = sp_io.loadmat('./test_data/maxwell_ref.mat')
        self.eq = equation.Equation( \
                int(mesh['N']),
                np.squeeze(mesh['VX']),
                np.squeeze(mesh['VY']),
                mesh['EToV'])

    def test_f_mask(self):
        """Checks if the face mask is implemented correctly."""
        with self.subTest('ManufacturedSolution'):
            r = np.array([0.0, 0.3, -1.0, 0.2, -0.6, -1.0, 0.5])
            s = np.array([-1.0, -0.3, -1.0, 0.7, 0.6, 0.0, 0.5])
            f_mask = equation.Equation.get_face_mask(r, s)

            expected = np.array([[0, 2], [1, 4], [2, 5]]).transpose()
            np.testing.assert_array_equal(expected, f_mask)

        with self.subTest('CompareWithRefFMask'):
            np.testing.assert_array_almost_equal( \
                    self.ref['Fmask'], self.eq.f_mask)

    def test_global_mesh_coordinates(self):
        """Checks if the global mesh is computed correctly."""
        with self.subTest('X'):
            np.testing.assert_array_almost_equal( \
                    self.ref['x'], self.eq.x, decimal=4)

        with self.subTest('Y'):
            np.testing.assert_array_almost_equal( \
                    self.ref['y'], self.eq.y, decimal=4)

    def test_metric_elements_for_local_mapping(self):
        """Checks if the metric elements are computed correctly."""
        with self.subTest('Rx'):
            np.testing.assert_array_almost_equal( \
                    self.ref['rx'], self.eq.r_x, decimal=2)

        with self.subTest('Ry'):
            np.testing.assert_array_almost_equal( \
                    self.ref['ry'], self.eq.r_y, decimal=2)

        with self.subTest('Sx'):
            np.testing.assert_array_almost_equal( \
                    self.ref['sx'], self.eq.s_x, decimal=2)

        with self.subTest('Sy'):
            np.testing.assert_array_almost_equal( \
                    self.ref['sy'], self.eq.s_y, decimal=2)

    def test_normal(self):
        """Checks if outward facing normals are computed correctly."""
        with self.subTest('Nx'):
            np.testing.assert_array_almost_equal( \
                    self.ref['nx'], self.eq.n_x, decimal=3)

        with self.subTest('Ny'):
            np.testing.assert_array_almost_equal( \
                    self.ref['ny'], self.eq.n_y, decimal=3)

        with self.subTest('SJ'):
            np.testing.assert_array_almost_equal( \
                    self.ref['sJ'], self.eq.s_j, decimal=3)

    def test_connectivity_matrices(self):
        """Checks if the connectivity matrices are build correctly."""

        with self.subTest('EToE'):
            np.testing.assert_array_equal(self.ref['EToE'], self.eq._e_to_e)

        with self.subTest('EToF'):
            np.testing.assert_array_equal(self.ref['EToF'], self.eq._e_to_f)


    def test_build_map(self):
        """Checks if maps for face nodes are constructed correctly."""
        with self.subTest('VMapM'):
            np.testing.assert_array_equal( \
                    np.squeeze(self.ref['vmapM'] - 1), self.eq.v_map_m)

        with self.subTest('MapM'):
            np.testing.assert_array_equal( \
                    np.squeeze(self.ref['mapM'] - 1), self.eq.map_m)

        with self.subTest('VMapP'):
            np.testing.assert_array_equal( \
                    np.squeeze(self.ref['vmapP'] - 1), self.eq.v_map_p)

        with self.subTest('MapP'):
            np.testing.assert_array_equal( \
                    np.squeeze(self.ref['mapP'] - 1), self.eq.map_p)

        with self.subTest('VMapB'):
            np.testing.assert_array_equal( \
                    np.squeeze(self.ref['vmapB'] - 1), self.eq.v_map_b)

        with self.subTest('MapB'):
            np.testing.assert_array_equal( \
                    np.squeeze(self.ref['mapB'] - 1), self.eq.map_b)

    def test_lift_matrix(self):
        """Checks if the LIFT matrix is computed correctly."""
        np.testing.assert_array_almost_equal( \
               self.ref['LIFT'], self.eq.lift, decimal=1)


if __name__ == '__main__':
    unittest.main()
