from collections import namedtuple

import numpy as np

from one_d import numerics_1d

# Tolerance for comparisons with 0.
TOL = 1e-12
# The floating point number data type.
DTYPE = np.float64
# Data structure for the cubature rule.
Cubature = namedtuple('Cubature', ('R', 'S', 'W', 'N'))

def divide_no_nan(a, b):
    """Performs a / b and returns 0 if b is 0."""
    b_new = np.where(np.abs(b) < TOL, np.ones_like(b), b)
    return np.where(np.abs(b) < TOL, np.zeros_like(a), a / b_new)


def simplex_2d_p(a, b, i, j):
    """Evaluates 2D orthonormal polynomial on simplex at (a, b).

    Args:
      a: The first simplex coordinate.
      b: The second simplex coordinate.
      i: The order of the first coordinate.
      j: The order of the second coordinate.

    Returns:
      The 2D orthonormal polynomial.
    """
    p_1 = numerics_1d.jacobi_p(a, 0, 0, i)
    p_2 = numerics_1d.jacobi_p(b, 2 * i + 1, 0, j)
    return np.sqrt(2.0) * p_1 * p_2 * (1.0 - b)**i


def grad_simplex_2d_p(a, b, i, j):
    """Computes the derivatives of the modal basis (i, j) on 2D simplex."""
    f_a = numerics_1d.jacobi_p(a, 0, 0, i)
    d_f_a = np.squeeze(numerics_1d.grad_jacobi_p(a, 0, 0, i))
    g_b = numerics_1d.jacobi_p(b, 2 * i + 1, 0, j)
    d_g_b = np.squeeze(numerics_1d.grad_jacobi_p(b, 2 * i + 1, 0, j))

    # Compute the r derivative:
    # d/dr = da /dr d/da + db/dr d/db = (2 / (1 - s)) d/da
    # = (2 / (1 - b)) d/da.
    d_mode_dr = d_f_a * g_b
    if i > 0:
        d_mode_dr *= (0.5 * (1 - b))**(i - 1)

    # Compute the s derivative:
    # d/ds = ((1 + a) / 2) / ((1 - b) / 2) d/da + d/db.
    d_mode_ds = d_f_a * (g_b * (0.5 * (1 + a)))
    if i > 0:
        d_mode_ds *= (0.5 * (1 - b))**(i - 1)

    tmp = d_g_b * ((0.5 * (1 - b))**i)
    if i > 0:
        tmp -= 0.5 * i * g_b * (0.5 * (1 - b))**(i - 1)

    d_mode_ds += f_a * tmp

    # Normalize.
    d_mode_dr *= 2**(i + 0.5)
    d_mode_ds *= 2**(i + 0.5)
    return d_mode_dr, d_mode_ds


def rs_to_ab(r, s):
    """Transfers from (r, s) to (a, b) coordinates in triangle."""
    a = np.where( \
            np.abs(s - 1.0) < TOL, \
            -1.0 * np.ones_like(r), \
            2.0 * divide_no_nan(1.0 + r, 1.0 - s) - 1)
    b = s
    return a, b


def xy_to_rs(x, y):
    """Converts (x, y) in equidistant to (r, s) in standard triangle."""
    l_1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    l_2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    l_3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0

    r = -l_2 + l_3 - l_1
    s = -l_2 - l_3 + l_1
    return r, s


def warp_factor(n, r_out):
    """Computes scaled warp function at order `n` on `r_our` nodes."""
    # Compute LGL and equidistant node distribution.
    r_lgl = numerics_1d.jacobi_gl(0, 0, n)
    r_eq = np.linspace(-1, 1, n + 1)

    # Compute the Vandermonde matrix based on the equidistant nodes.
    v_eq = numerics_1d.vandermonde_1d(n, r_eq)

    # Evaluate Lagrange polynomials at r_out.
    n_r = len(r_out)
    p_mat = np.stack( \
            [np.squeeze(numerics_1d.jacobi_p(r_out, 0, 0, i)) \
            for i in range(n + 1)])
    l_mat = np.linalg.solve(v_eq.T, p_mat)

    # Compute the warp factor.
    warp = np.matmul(l_mat.T, r_lgl - r_eq)

    # Scale the warp factor.
    zerof = np.where( \
            np.abs(r_out) < 1.0 - 1e-10, \
            np.ones_like(r_out), \
            np.zeros_like(r_out))
    sf = 1.0 - (zerof * r_out)**2
    return divide_no_nan(warp, sf) + warp * (zerof - 1.0)


def nodes_2d(n):
    """Computes nodes in equilateral triangle for polynomial order `n`."""
    alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, \
            1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]

    # Set optimized parameter, alpha, depending on order `n`.
    alpha = alpha_opt[n] if n < 16 else 5.0 / 3.0

    # Total number of nodes.
    n_p = int((n + 1) * (n + 2) / 2)

    # Create equidistributed nodes on equilateral triagnle.
    l_1 = np.zeros((n_p,), dtype=DTYPE)
    l_3 = np.zeros((n_p,), dtype=DTYPE)
    sk = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            l_1[sk] = i / n
            l_3[sk] = j / n
            sk += 1
    l_2 = 1.0 - l_1 - l_3
    x = -l_2 + l_3
    y = (-l_2 - l_3 + 2.0 * l_1) / np.sqrt(3.0)

    # Compute blending function at each node for each edge.
    blend_1 = 4.0 * l_2 * l_3
    blend_2 = 4.0 * l_1 * l_3
    blend_3 = 4.0 * l_1 * l_2

    # Amount of warp for each node, for each edge.
    warp_1 = warp_factor(n, l_3 - l_2)
    warp_2 = warp_factor(n, l_1 - l_3)
    warp_3 = warp_factor(n, l_2 - l_1)

    # Combine blend and warp.
    def blend_warp(blend, warp, l):
        """Combines the blend and warp."""
        return blend * warp * (1.0 + (alpha * l)**2)

    warp_1 = blend_warp(blend_1, warp_1, l_1)
    warp_2 = blend_warp(blend_2, warp_2, l_2)
    warp_3 = blend_warp(blend_3, warp_3, l_3)

    # Accumulate deformations associated with each edge.
    x += warp_1 + np.cos(2.0 * np.pi / 3.0) * warp_2 + \
            np.cos(4.0 * np.pi / 3.0) * warp_3
    y += np.sin(2.0 * np.pi / 3.0) * warp_2 + \
            np.sin(4.0 * np.pi / 3.0) * warp_3

    return x, y


def vandermonde_2d(n, r, s):
    """Initializes the 2D Vandermonde matrix V_{ij} = phi_j(r_i, s_i)."""
    # Transfer to (a, b) coordinates.
    a, b = rs_to_ab(r, s)

    # Build the Vandermonde matrix.
    n_p = int((n + 1) * (n + 2) / 2)
    v_2d = np.zeros((len(r), n_p), dtype=DTYPE)
    sk = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            v_2d[:, sk] = simplex_2d_p(a, b, i, j)
            sk += 1

    return v_2d


def grad_vandermonde_2d(n, r, s):
    """Initializes the gradient of the modal basis at (r, s) at order n."""
    n_p = int((n + 1) * (n + 2) / 2)
    v_2d_r = np.zeros((len(r), n_p), dtype=DTYPE)
    v_2d_s = np.zeros((len(r), n_p), dtype=DTYPE)

    # Find the tensor-product coordinates.
    a, b = rs_to_ab(r, s)

    # Initialize matrices.
    sk = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            v_2d_r[:, sk], v_2d_s[:, sk] = grad_simplex_2d_p(a, b, i, j)
            sk += 1

    return v_2d_r, v_2d_s


def d_matrices_2d(n, r, s, v):
    """Initializes the (r, s) differentiation on the simplex at order n."""
    v_r, v_s = grad_vandermonde_2d(n, r, s)
    inv_v = np.linalg.inv(v)
    return np.matmul(v_r, inv_v), np.matmul(v_s, inv_v)


def filter_2d(n, n_c, s, v, inv_v=None):
    """Initializes 2D filter matrix of order s and cutoff n_c."""
    assert(s % 2 == 0)

    if inv_v is None:
        inv_v = np.linalg.inv(v)

    v_f = np.copy(v)
    eps = np.finfo(float).resolution
    alpha = -np.log(eps)

    # Compute the product of V F.
    sk = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            if i + j >= n_c:
                filter_diag = np.exp( \
                        -alpha * (float(i + j - n_c) / float(n - n_c))**s)

                v_f[:, sk] *= filter_diag
            sk += 1

    return np.matmul(v_f, inv_v)


def cubature_2d(c_order):
    """Provides cubature rules to integrate up to `c_order` polynomial."""
    data = np.load('two_d/data/cubature.npy', allow_pickle=True)[0]

    if c_order <= 28:
        cub = Cubature(
                R=data[c_order - 1][:, 0],
                S=data[c_order - 1][:, 1],
                W=data[c_order - 1][:, 2],
                N=len(data[c_order - 1][:, 2]))
    else:
        n = int(np.ceil((c_order + 1.0) / 2.0))
        cub_a, cub_wa = numerics_1d.jacobi_gq(0, 0, n - 1)
        cub_b, cub_wb = numerics_1d.jacobi_gq(1, 0, n - 1)
        print(cub_b)
        print(cub_wb)

        cub_a = np.tile(np.reshape(cub_a, (1, len(cub_a))), (n, 1))
        cub_b = np.tile(np.reshape(cub_b, (len(cub_b), 1)), (1, n))

        cub_r = 0.5 * (1.0 + cub_a) * (1.0 - cub_b) - 1.0
        cub_s = cub_b
        cub_w = 0.5 * np.matmul(
                np.reshape(cub_wb, (len(cub_wb), 1)),
                np.reshape(cub_wa, (1, len(cub_wa))))

        n_cub = np.prod(cub_w.shape)
        cub = Cubature(
                R=np.reshape(cub_r.T, (n_cub,)),
                S=np.reshape(cub_s.T, (n_cub,)),
                W=np.reshape(cub_w.T, (n_cub,)),
                N=n_cub)

    return cub
