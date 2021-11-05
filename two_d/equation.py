import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import numpy as np

from one_d import numerics_1d
from two_d import numerics_2d

# The tolerance for considering a node as a specific type of node.
NODE_TOL = 1e-6


class Equation(object):
    """A library of helper variables and functions for 2D DG method."""

    def __init__(self, n, vx, vy, e_to_v):
        """Initializes the 2D equation library.

        Args:
          n: The order of the polynomial.
          vx: The vortex coordinates in the x direction of length K + 1.
          vy: The vortex coordinates in the y direction of length K + 1.
          e_to_v: A K by 3 matrix. Each row of the matrix specifies the
            3 vortices of a triangular element.
        """
        self._n = n
        self._n_p = int((n + 1) * (n + 2) / 2)
        self._n_faces = 3
        self._n_fp = n + 1

        self._e_to_v = e_to_v

        # Get the local mesh metrics for 1 triangular element.
        x, y = numerics_2d.nodes_2d(n)
        self.r, self.s = numerics_2d.xy_to_rs(x, y)

        # Get the global mesh coordinates for all nodes in all elements.
        self.x = self._get_global_coordinates(vx, self.r, self.s)
        self.y = self._get_global_coordinates(vy, self.r, self.s)

        # Get the 2D Vandermonde matrix and the derivative matrix.
        self.v = vandermonde_2d(n, self.r, self.s)
        self.inv_v = np.linalg.inv(self.v)
        self.d_r, self.d_s = d_matrices_2d(n, self.r, self.s, self.v)

        # Get the metric elements for the local mappings of the elements.
        self.r_x, self.s_x, self.r_y, self.s_y, self.jac = \
            get_geometric_factors(self.x, self.y, self.d_r, self.d_s)

        # Get the mask for all faces. The size of the mask is (n + 1) x 3.
        self.f_mask = get_face_mask(self.r, self.s)
        f_mask = flatten(self.f_mask)
        self.f_x = self.x[f_mask, :]
        self.f_y = self.y[f_mask, :]

        # Get the lift matrix.
        self.lift = self._lift_fn()

    @property
    def n_p(self):
        """Retrieves the number of quadrature points."""
        return self._n_p

    @property
    def n_faces(self):
        """Retrieves the number of quadrature points."""
        return self._n_faces

    @property
    def n_fp(self):
        """Retrieves the number of face points."""
        return self._n_fp

    @staticmethod
    def flatten(u):
        """Reshapes `u` into a 1D array."""
        return np.reshape(u, (np.prod(u.shape),))

    @staticmethod
    def expand(u, n_0, n_1):
        """Reshapes `u` into a 2D array."""
        return np.reshape(u, (n_0, n_1))

    @staticmethod
    def get_face_mask(r, s):
        """Identifies nodes on faces of a triangular element."""
        f_mask_1 = np.where(np.abs(s + 1.0) < NODE_TOL)
        f_mask_2 = np.where(np.abs(r + s) < NODE_TOL)
        f_mask_3 = np.where(np.abs(r + 1.0) < NODE_TOL)
        return np.concatenate([f_mask_1, f_mask_2, f_mask_3]).transpose()

    @staticmethod
    def get_geometric_factors(x, y, d_r, d_s):
        """Computes the metric elements for local mappings of elements."""
        x_r = np.matmul(d_r, x)
        x_s = np.matmul(d_s, x)
        y_r = np.matmul(d_r, y)
        y_s = np.matmul(d_s, y)
        jac = -x_s * y_r + x_r * y_s
        r_x = y_s / jac
        s_x = -y_r / jac
        r_y = -x_s / jac
        s_y = x_r / jac
        return r_x, s_x, r_y, s_y, jac

    @staticmethod
    def normals(x, y, d_r, d_s):
        """Computes outward pointing normals at faces and surface Jacobian."""
        x_r = np.matmul(d_r, x)
        x_s = np.matmul(d_s, x)
        y_r = np.matmul(d_r, y)
        y_s = np.matmul(d_s, y)
        jac = -x_s * y_r + x_r * y_s


    def _get_global_coordinates(self, v, r, s):
        """Computes the global coordinates for all nodes in all elements."""
        return 0.5 * (-(r + s) * v[self._e_to_v[:, 0]] +
                      (1.0 + r) * v[self._e_to_v[:, 1]] +
                      (1.0 + s) * v[self._e_to_v[:, 2]])

    def _lift_fn(self):
        """Computes surface to volume lift term for DG formulation."""
        e_mat = np.zeros((self.n_p, self.n_faces * self.n_fp))

        def get_mass_edge(i):
            """Computes the mass matrix at face/edge `i`."""
            l = self.s if i == 2 else self.r
            face_l = l[self.f_mask[:, i]]
            v_1d = numerics_1d.vandermonde_1d(self._n, face_l)
            return np.linalg.inv(np.matmul(v_1d, v_1d.transpose()))

        for i in range(3):
            e_mat[f_mask[:, i], i * self.n_fp:(i + 1) * self.n_fp] = \
                get_mass_edge(i)

        # inv(mass matrix) * \I_n (L_i, L_j)_{edge_n}.
        return np.matmul(self.v, np.matmul(self.v.transpose(), e_mat))

    def grad(self, u):
        """Computes the 2D gradient field of scalar `u`."""
        u_r = np.matmul(self.d_r, u)
        u_s = np.matmul(self.d_s, u)
        u_x = self.r_x * u_r + self.s_x * u_s
        u_y = self.r_y * u_r + self.s_y * u_s
        return u_x, u_y

    def div(self, u, v):
        """Computes the 2D divergence of vector field (u, v)."""
        u_r = np.matmul(self.d_r, u)
        u_s = np.matmul(self.d_s, u)
        v_r = np.matmul(self.d_r, v)
        v_s = np.matmul(self.d_s, v)
        return self.r_x * u_r + self.s_x * u_s + self.r_y * v_r + \
            self.s_y * v_s

    def curl(self, u, v, w=None):
        """Computes the 2D curl operator in the (x, y) plane."""
        u_r = np.matmul(self.d_r, u)
        u_s = np.matmul(self.d_s, u)
        v_r = np.matmul(self.d_r, v)
        v_s = np.matmul(self.d_s, v)
        omg_x = None
        omg_y = None
        omg_z = self.r_x * v_r + self.s_x * v_s - self.r_y * u_r - \
            self.s_y * u_s

        if w is not None:
            w_r = np.matmul(self.d_r, w)
            w_s = np.matmul(self.d_s, w)
            omg_x = self.r_y * w_r + self.s_y * w_s
            omg_y = -self.r_x * w_r - self.s_x * w_s

        return omg_x, omg_y, omg_z
