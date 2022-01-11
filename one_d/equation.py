"""A library for solving the 1D linear conservation equation."""
import numpy as np

from one_d import geometry_1d
from one_d import numerics_1d

_NODE_TOL = 1e-10


class Equation(object):
    """A library that defines operators in the linear conservation eq."""

    def __init__(self, n, vx, e_to_v):
        """Initializes the 1D equation library.

        Args:
          n: The order of the polynomial.
          vx: The vortex coordinates of length K + 1.
          e_to_v: A K by 2 matrix. Each row of the matrix specifies the
            start and end points of an element.
        """
        self._n = n
        self._n_p = n + 1
        self._n_faces = 2
        self._n_fp = 1

        self.r = numerics_1d.jacobi_gl(0, 0, n)
        self.v = numerics_1d.vandermonde_1d(n, self.r)
        self.dr = numerics_1d.d_matrix_1d(n, self.r, self.v)

        self.inv_v = np.linalg.inv(self.v)

        # Get the coordinate information.
        self.k = len(vx) - 1
        assert(e_to_v.shape[0] == self.k)

        def affine_mapping(x_0, x_1):
            """Maps the LGL nodes `r` to physical coordinates."""
            return x_0 + 0.5 * (self.r + 1) * (x_1 - x_0)

        x = [affine_mapping(vx[e_to_v[i, 0]], vx[e_to_v[i, 1]]) \
                for i in range(self.k)]
        self.x = np.transpose(np.array(x))

        # Compute the geometric factors.
        # J: the local transformation Jacobian.
        self.j = np.matmul(self.dr, self.x)
        self.rx = 1.0 / self.j

        # Compute masks for the edge nodes.
        f_mask_0 = np.where(np.abs(self.r + 1) < _NODE_TOL)[0][0]
        f_mask_1 = np.where(np.abs(self.r - 1) < _NODE_TOL)[0][0]
        f_mask = [f_mask_0, f_mask_1]
        self.f_x = np.stack(
                [self.x[f_mask_0, :], self.x[f_mask_1, :]], axis=1)

        # Build surface normals and inverse metric at surface
        self.n_x = np.zeros((self._n_fp * self._n_faces, self.k))
        self.n_x[0, :] = -1.0
        self.n_x[1, :] = 1.0
        self.f_scale = np.stack(
                [1.0 / self.j[f_mask_0, :], 1.0 / self.j[f_mask_1, :]], \
                        axis=1).T

        # Build connectivity matrix.
        [self.e_to_e, self.e_to_f] = geometry_1d.connect_1d(e_to_v)

        # Build connectivity map.
        def build_maps_1d():
            """Builds the connectivity and boundary tables."""
            x = np.reshape(self.x, (np.prod(self.x.shape),))

            node_ids = np.reshape( \
                    np.arange(self.k * self._n_p), (self._n_p, self.k))
            v_map_m = np.zeros((self._n_fp, self._n_faces, self.k))
            v_map_p = np.zeros((self._n_fp, self._n_faces, self.k))
            for k1 in range(self.k):
                for f1 in range(self._n_faces):
                    # Find index of face nodes wrt volume node ordering.
                    v_map_m[:, f1, k1] = node_ids[f_mask[f1], k1]

            for k1 in range(self.k):
                for f1 in range(self._n_faces):
                    # Find neighbor.
                    k2 = int(self.e_to_e[k1, f1])
                    f2 = int(self.e_to_f[k1, f1])

                    # Find volume node numbers of left and right nodes.
                    v_id_m = int(v_map_m[:, f1, k1])
                    v_id_p = int(v_map_m[:, f2, k2])

                    x1 = x[v_id_m]
                    x2 = x[v_id_p]

                    # Compute distance matrix.
                    d = (x1 - x2)**2
                    if d < _NODE_TOL:
                        v_map_p[:, f1, k1] = v_id_p

            v_map_m = np.reshape(v_map_m, (np.prod(v_map_m.shape),))
            v_map_p = np.reshape(v_map_p, (np.prod(v_map_p.shape),))

            # Create list of boundary nodes.
            map_b = np.array(np.where(v_map_p == v_map_m))
            v_map_b = np.array(v_map_m[map_b])

            return v_map_m.astype(np.int32), v_map_p.astype(np.int32), \
                    v_map_b.astype(np.int32), map_b.astype(np.int32)

        self.v_map_m, self.v_map_p, self.v_map_b, self.map_b = \
                build_maps_1d()
        self.map_i = 0
        self.map_o = self.k * self._n_faces - 1
        self.v_map_i = 0
        self.v_map_o = self.k * self._n_p - 1

        self.lift = self.lift_fn()

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

    def lift_fn(self):
        """Computs the surface integral term in the DG formulation."""
        e_mat = np.zeros((self._n_p, self._n_faces * self._n_fp))
        e_mat[0, 0] = 1.0
        e_mat[self._n_p - 1, 1] = 1.0
        return np.matmul(self.v, np.matmul(self.v.T, e_mat))

    def flatten(self, u):
        """Reshapes `u` into a 1D array."""
        return np.reshape(u, (np.prod(u.shape),))

    def expand(self, u, n_0, n_1):
        """Reshapes `u` into a 2D array."""
        return np.reshape(u, (n_0, n_1))

    def filter(self, n_c, s):
        """Initializes a 1D filter matrix."""
        assert(s % 2 == 0)

        v_f = np.copy(self.v)
        eps = np.finfo(float).resolution
        alpha = -np.log(eps)

        for i in range(n_c, self._n):
            filter_diag = np.exp(\
                    -alpha * ((i + 1 - n_c) / (self._n - n_c))**s)
            v_f[:, i] *= filter_diag

        return np.matmul(v_f, self.inv_v)

    def slope_limit_1(self, u):
        """Applies slope limiter Pi^1 to `u`."""
        # Compute modal coefficients.
        uh = np.matmul(self.inv_v, u)

        # Extract the linear polynomial.
        u_l = np.copy(uh)
        u_l[2:] = 0.0
        u_l = np.matmul(self.v, u_l)

        # Find the gradient of the linear polynomial.
        h = np.tile(self.x[-1, :] - self.x[0, :], (self.n_p, 1))
        u_x = (2.0 / h) * np.matmul(self.dr, u_l)

        # Extract cell avereages.
        uh[1:, :] = 0.0
        u_avg = np.matmul(self.v, uh)
        v = u_avg[0, :]

        # Find cell averages in neighborhood of each element.
        vk = v
        vk_m1 = np.concatenate([[v[0]], v[:-1]], axis=0)
        vk_p1 = np.concatenate([v[1:], [v[-1]]], axis=0)

        # Limit function in all cells.
        return numerics_1d.slope_limit_lin(u_x, self.x, vk_m1, vk, vk_p1)

    def slope_limit_n(self, u):
        """Applies slope limiter Pi^N to an N'th order polynomial `u`."""
        # Compute cell averages.
        uh = np.matmul(self.inv_v, u)
        uh[1:, :] = 0.0
        u_avg = np.matmul(self.v, uh)
        v = u_avg[0, :]
        vk = v
        vk_m1 = np.concatenate([[v[0]], v[:-1]], axis=0)
        vk_p1 = np.concatenate([v[1:], [v[-1]]], axis=0)

        # Apply slope limiter as needed.
        u_limit = u
        eps_0 = 1e-8

        # Find end values of each element.
        u_e1 = u[0, :]
        u_e2 = u[-1, :]

        # Apply reconstruction to find elements in need of limiting.
        v_e1 = vk - numerics_1d.minmod( \
                np.array([vk - u_e1, vk - vk_m1, vk_p1 - vk]))
        v_e2 = vk + numerics_1d.minmod( \
                np.array([u_e2 - vk, vk - vk_m1, vk_p1 - vk]))
        ids = np.where(np.logical_or(np.abs(v_e1 - u_e1) > eps_0, \
                np.abs(v_e2 - u_e2) > eps_0))[0]

        # Check to see if any elements require limiting.
        if len(ids) != 0:
            # Create piecewise linear solution for limiting on specified
            # elements.
            uh_l = np.einsum('ij,jk->ik', self.inv_v, u[:, ids])
            uh_l[2:, :] = 0.0
            u_l = np.matmul(self.v, uh_l)

            # Find the gradient of the linear polynomial.
            h = np.tile( \
                    self.x[-1, ids] - self.x[0, ids], (self.n_p, 1))
            u_x = (2.0 / h) * np.matmul(self.dr, u_l)

            # Apply slope limiter to selected elements.
            u_limit[:, ids] = numerics_1d.slope_limit_lin( \
                    u_x, self.x[:, ids], vk_m1[ids], vk[ids], vk_p1[ids])

        return u_limit
