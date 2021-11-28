"""A library for 2D equation discretizations with nodal DG."""

import enum
import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import numpy as np

from one_d import numerics_1d
from two_d import geometry
from two_d import numerics_2d

# The tolerance for considering a node as a specific type of node.
NODE_TOL = 1e-12
# The floating point number data type.
DTYPE = np.float64


class BoundaryCondition(enum.Enum):
    """Defines the type of boundary condition."""
    IN = 1
    OUT = 2
    WALL = 3
    FAR = 4
    CYL = 5
    DIRICHLET = 6
    NEUMANN = 7
    SLIP = 8


class Equation(object):
    """A library of helper variables and functions for 2D DG method."""

    def __init__(self, n, vx, vy, e_to_v):
        """Initializes the 2D equation library.

        Args:
          n: The order of the polynomial.
          vx: The vortex coordinates in the x direction.
          vy: The vortex coordinates in the y direction.
          e_to_v: A K by 3 matrix. Each row of the matrix specifies the
            3 vortices of a triangular element.
        """
        self._n = n
        self._n_p = int((n + 1) * (n + 2) / 2)
        self._n_faces = 3
        self._n_fp = n + 1
        self._k, _ = e_to_v.shape

        self._vx = vx
        self._vy = vy

        # Get the connectivity matrices for elements, faces, and nodes.
        self._e_to_v = e_to_v
        self._e_to_e, self._e_to_f = geometry.connect(self._e_to_v)

        # Get the local mesh metrics for 1 triangular element.
        x, y = numerics_2d.nodes_2d(n)
        self.r, self.s = numerics_2d.xy_to_rs(x, y)

        # Get the global mesh coordinates for all nodes in all elements.
        self.x = self._get_global_coordinates(vx, self.r, self.s)
        self.y = self._get_global_coordinates(vy, self.r, self.s)

        # Get the 2D Vandermonde matrix.
        self.v = numerics_2d.vandermonde_2d(n, self.r, self.s)
        self.inv_v = np.linalg.inv(self.v)

        # Compute the mass matrix.
        self.m = np.matmul(self.inv_v.T, self.inv_v)

        # Compute the derivative matrix in strong form.
        self.d_r, self.d_s = numerics_2d.d_matrices_2d( \
                n, self.r, self.s, self.v)

        # Compute the derivative matrix in weak form.
        self.v_r, self.v_s = numerics_2d.grad_vandermonde_2d( \
                self._n, self.r, self.s)
        inv_v2 = np.linalg.inv(np.matmul(self.v, self.v.T))
        self.d_rw = np.matmul(np.matmul(self.v, self.v_r.T), inv_v2)
        self.d_sw = np.matmul(np.matmul(self.v, self.v_s.T), inv_v2)

        # Get geometric factors for the local mappings of the elements.
        self.r_x, self.s_x, self.r_y, self.s_y, self.jac = \
            self.get_geometric_factors(self.x, self.y, self.d_r, self.d_s)

        # Get the mask for all faces. The size of the mask is (n + 1) x 3.
        self.f_mask = self.get_face_mask(self.r, self.s)
        f_mask = self.flatten(self.f_mask)
        self.f_x = self.x[f_mask, :]
        self.f_y = self.y[f_mask, :]

        # Get the outward pointing normals at faces.
        self.n_x, self.n_y, self.s_j = self._normals()
        self.f_scale = self.s_j / self.jac[f_mask, :]

        # Get the connectivity and boundary tables in the K number of np
        # elements.
        self.v_map_m, self.map_m, self.v_map_p, self.map_p, self.v_map_b, \
                self.map_b = self._build_map()

        # Get the lift matrix, i.e. the surface integral term.
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
    def expand(u, n_0, n_1):
        """Reshapes `u` into a 2D array following a column first order."""
        return np.reshape(u, (n_1, n_0)).T

    @staticmethod
    def flatten(u):
        """Reshapes `u` into a 1D array, a reverse function of expand."""
        dims = len(u.shape)
        return np.reshape( \
                u.transpose(np.arange(dims)[::-1]), (np.prod(u.shape),))

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

    def _normals(self):
        """Computes outward pointing normals at faces and surface Jacobian."""
        x_r = np.matmul(self.d_r, self.x)
        x_s = np.matmul(self.d_s, self.x)
        y_r = np.matmul(self.d_r, self.y)
        y_s = np.matmul(self.d_s, self.y)
        jac = -x_s * y_r + x_r * y_s

        # Interpolate geometric factors to face nodes.
        f_mask = self.flatten(self.f_mask)
        f_xr = x_r[f_mask, :]
        f_xs = x_s[f_mask, :]
        f_yr = y_r[f_mask, :]
        f_ys = y_s[f_mask, :]

        # Build normals.
        n_x = np.zeros((3 * self._n_fp, self._k), dtype=DTYPE)
        n_y = np.zeros((3 * self._n_fp, self._k), dtype=DTYPE)
        f_id1 = np.arange(self._n_fp)
        f_id2 = np.arange(self._n_fp, int(2 * self._n_fp))
        f_id3 = np.arange(int(2 * self._n_fp), int(3 * self._n_fp))

        # Face 1.
        n_x[f_id1, :] = f_yr[f_id1, :]
        n_y[f_id1, :] = -f_xr[f_id1, :]

        # Face 2.
        n_x[f_id2, :] = f_ys[f_id2, :] - f_yr[f_id2, :]
        n_y[f_id2, :] = -f_xs[f_id2, :] + f_xr[f_id2, :]

        # Face 3.
        n_x[f_id3, :] = -f_ys[f_id3, :]
        n_y[f_id3, :] = f_xs[f_id3, :]

        # Normalize the normal vectors.
        s_j = np.sqrt(n_x**2 + n_y**2)
        n_x /= s_j
        n_y /= s_j

        return n_x, n_y, s_j

    def _build_map(self):
        """Constructs connectivity and boundary tables for all elements.

        Returns:
          v_map_m: The indices of interior face nodes for all faces.
          map_m: The global indices for nodes in v_map_m.
          v_map_p: The indices of exterior face nodes for all faces.
          map_p: The global indices for nodes in v_map_p.
          v_map_b: The indices of boundary face nodes for all faces.
          map_b: The global indices for nodes in v_map_b.
        """
        node_ids = self.expand( \
                np.arange(self._k * self._n_p), self._n_p, self._k)
        v_map_p = np.zeros((self._n_fp, self._n_faces, self._k))
        map_m = np.arange(self._k * self._n_fp * self._n_faces)
        map_p = np.reshape( \
                np.copy(map_m), (self._k, self._n_faces, self._n_fp) \
                ).transpose(2, 1, 0)

        # Find indices of face nodes wrt volume node ordering.
        # Get indices corresponds to interior nodes.
        v_map_m = np.array([ \
                [node_ids[self.f_mask[:, f], k] for f in range(self._n_faces)] \
                for k in range(self._k)]).transpose((2, 1, 0))

        x = self.flatten(self.x)
        y = self.flatten(self.y)

        # Get indices corresponds to exterior nodes.
        for k1 in range(self._k):
            for f1 in range(self._n_faces):
                # Find the neibhoring element and face.
                k2 = self._e_to_e[k1, f1]
                f2 = self._e_to_f[k1, f1]

                # Find the reference length of the edge.
                v1 = self._e_to_v[k1, f1]
                v2 = self._e_to_v[k1, (f1 + 1) % self._n_faces]
                d_ref = np.sqrt( \
                        (self._vx[v1] - self._vx[v2])**2 + \
                        (self._vy[v1] - self._vy[v2])**2)

                # Find the volume node numbers of left and right nodes.
                v_id_m = v_map_m[:, f1, k1]
                v_id_p = v_map_m[:, f2, k2]
                x1 = np.reshape(x[v_id_m], (len(x[v_id_m]), 1))
                y1 = np.reshape(y[v_id_m], (len(y[v_id_m]), 1))
                x2 = np.reshape(x[v_id_p], (1, len(x[v_id_p])))
                y2 = np.reshape(y[v_id_p], (1, len(y[v_id_p])))

                # Find the distance between each point on face 1 to each point
                # on face 2.
                d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                # Find the exterior nodes for all interior nodes on the present
                # face.
                id_m, id_p = np.where(d < NODE_TOL * d_ref)
                v_map_p[id_m, f1, k1] = v_id_p[id_p]
                map_p[id_m, f1, k1] = \
                        id_p + f2 * self._n_fp + k2 * self._n_faces * self._n_fp

        # Reshape v_map_m and v_map_b to vectors, and create boundary node list.
        v_map_m = self.flatten(v_map_m)
        v_map_p = self.flatten(v_map_p)
        map_p = self.flatten(map_p)
        map_b = np.squeeze(np.where(v_map_p == v_map_m))
        v_map_b = v_map_m[map_b]

        return v_map_m, map_m, v_map_p, map_p, v_map_b, map_b

    def _get_global_coordinates(self, v, r, s):
        """Computes the global coordinates for all nodes in all elements."""

        return 0.5 * (-np.outer(r + s, v[self._e_to_v[:, 0]]) + \
                np.outer(1.0 + r, v[self._e_to_v[:, 1]]) + \
                np.outer(1.0 + s, v[self._e_to_v[:, 2]]))

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
            e_mat[self.f_mask[:, i], i * self.n_fp:(i + 1) * self.n_fp] = \
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
        return self.r_x * u_r + self.s_x * u_s + self.r_y * v_r + self.s_y * v_s

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

    def dt_scale(self):
        """Computes inscribed circle diameter for grid to choose timestep."""
        # Find vertex nodes.
        v_mask = self.flatten(np.array([
                np.where(np.abs(self.s + self.r + 2.0) < NODE_TOL),
                np.where(np.abs(self.r - 1.0) < NODE_TOL),
                np.where(np.abs(self.s - 1.0) < NODE_TOL),
                ]))
        vx = self.x[v_mask, :]
        vy = self.y[v_mask, :]

        # Compute semi-perimeter and area.
        l = np.array([np.sqrt((vx[i, :] - vx[(i + 1) % 3, :])**2 + \
                (vy[i, :] - vy[(i + 1) % 3, :])**2) for i in range(3)])
        sper = np.sum(l, 0) / 2.0
        area = np.sqrt(sper * (sper - l[0]) * (sper - l[1]) * (sper - l[2]))

        return area / sper
