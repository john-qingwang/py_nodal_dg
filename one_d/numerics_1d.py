"""A library for 1D nodal-DG numerics."""

import numpy as np
import scipy.special as ss

DTYPE = np.float64


def jacobi_gq(alpha, beta, n):
    """Computes the n'th order Gauss quadrature points and weights."""
    if n == 0:
        r = np.array(-(alpha - beta) / (alpha + beta + 2.0))
        w = np.array(2.0)
    else:
        h1 = 2.0 * np.arange(n + 1) + alpha + beta
        if alpha + beta < 10 * np.finfo(DTYPE).resolution:
            d0 = np.zeros((n + 1, n + 1), dtype=DTYPE)
        else:
            d0 = np.diag(-0.5 * (alpha**2 - beta**2) / (h1 + 2.0) / h1)
        j =  d0 + np.diag(2.0 / (h1[:-1] + 2.0) * np.sqrt( \
                        np.arange(1, n + 1) * \
                        (np.arange(1, n + 1) + alpha + beta) * \
                        (np.arange(1, n + 1) + alpha) * \
                        (np.arange(1, n + 1) + beta) / (h1[:-1] + 1.0) / \
                        (h1[:-1] + 3.0)), 1)
        j += j.T

        r, v = np.linalg.eig(j)
        w = v[0, :].T**2 * np.power(2.0, (alpha + beta + 1)) / \
            (alpha + beta + 1) * ss.gamma(alpha + 1) * ss.gamma(beta + 1) / \
            ss.gamma(alpha + beta + 1)

    return r, w


def jacobi_gl(alpha, beta, n):
    """Computes the n'th order Gauss lobatto quadrature points."""
    if n == 1:
        x = np.zeros((2,))
        x[0] = -1.0
        x[1] = 1.0
        return x

    x, _ = jacobi_gq(alpha + 1, beta + 1, n - 2)
    return np.concatenate([[-1.0], x, [1.0]])


def jacobi_p(x, alpha, beta, n):
    """Computes the Jacobi Polynomial.

    Args:
      x: Nodal points where the polynomial is computed.
      alpha: Parameter of the Jacobi polynomial, must be greeater than -1.
      beta: Parameter of the Jacobi polynomial, must be greeater than -1.
      n: Degree of the polynomial.

    Returns:
      The Jacobi polynomial at points `r` for order `n`.
    """
    p = np.zeros((len(x), 2))

    # Initial values of p_0(x) and p_1(x).
    gamma_0 = 2**(alpha + beta + 1) / (alpha + beta + 1) * \
            ss.gamma(alpha + 1) * ss.gamma(beta + 1) / \
            ss.gamma(alpha + beta + 1)
    p[:, 0] = 1.0 / np.sqrt(gamma_0)
    if n == 0:
        return p[:, 0]
    gamma_1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma_0
    p[:, 1] = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / \
            np.sqrt(gamma_1)
    if n == 1:
        return p[:, 1]

    a_old = 2.0 / (2.0 + alpha + beta) * \
            np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))

    for i in range(1, n):
        h_1 = 2 * i + alpha + beta
        a_new = 2.0 / (h_1 + 2) * np.sqrt((i + 1) * \
                (i + 1 + alpha + beta) * (i + 1 + alpha) * \
                (i + 1 + beta) / (h_1 + 1) / (h_1 + 3))
        b_new = -(alpha**2 - beta**2) / h_1 / (h_1 + 2)
        p_new = 1.0 / a_new * (-a_old * p[:, 0] + (x - b_new) * p[:, 1])
        p[:, 0] = p[:, 1]
        p[:, 1] = p_new
        a_old = a_new

    return p[:, 1]


def grad_jacobi_p(r, alpha, beta, n):
    """Computes the derivative of the Jacobi Polynomial.

    Args:
      r: Nodal points where the derivative is computed.
      alpha: Parameter of the Jacobi polynomial, must be greeater than -1.
      beta: Parameter of the Jacobi polynomial, must be greeater than -1.
      n: Degree of the polynomial.

    Returns:
      The derivative of the Jacobi polynomial at points `r` for order `n`.
    """
    if n < 0:
        raise ValueError(
            'Degree of the polynomial must be non-negative. '
            '{} is given.'.format(n))
    dp = np.zeros((len(r), 1))
    if n > 0:
        dp = (np.sqrt(n * (n + alpha + beta + 1.0)) *
                jacobi_p(r, alpha + 1, beta + 1, n - 1))
    return dp


def vandermonde_1d(n, r):
    """Initializes the 1D Vandermonde matrix, V_{ij} = phi_j(r_i)."""
    v = np.zeros((len(r), n + 1))
    for j in range(n + 1):
        v[:, j] = jacobi_p(r, 0, 0, j)
    return v


def grad_vandermonde_1d(n, r):
    """Initializes the gradient of the modal basis at `r` at order `n`."""
    d_vr = np.zeros((len(r), n + 1))
    for i in range(n + 1):
        d_vr[:, i] = np.squeeze(grad_jacobi_p(r, 0, 0, i))
    return d_vr


def d_matrix_1d(n, r, v):
    """Initializes the differentiation matrices on the interval.

    Args:
      n: The order of the polynomial.
      r: The nodal points.
      v: The Vandemonde matrix.

    Returns:
      The gradient matrix D.
    """
    vr = grad_vandermonde_1d(n, r)
    return np.linalg.lstsq(v.T, vr.T, rcond=None)[0].T


def minmod(v):
    """Computes the minmod function."""
    m, n = v.shape
    s = np.sum(np.sign(v), axis=0) / m
    return np.where(np.abs(s) == 1, s * np.min(np.abs(v), axis=0), 0)


def minmodb(v, max_dd2, h):
    """Computes the TVB modified minmod function.

    Args:
      v: A 2D array with shape (n, k) representing k vectors of length n.
      max_dd2: The upper bound on the second derivative at the local
        extrema.
      h: The grid spacing.

    Returns:
      The modified minmod function.
    """
    m_fn = v[0, :]
    ids = np.where(np.abs(m_fn) > max_dd2 * h**2)

    if len(ids[0]) > 0:
        m_fn[ids] = minmod(v[:, ids])

    return m_fn


def slope_limit_lin(u_x, x_l, v_m1, v_0, v_p1):
    """Applies slope limiter to a linear function u.

    Args:
      u_x: The gradient of a linear polynomial of size (n_p, 1).
      x_l: The collocation points of the linear element. The size of x_l is
        (n_p, 1).
      v_m1: The average of the left element.
      v_0: The average of the current element.
      v_p1: The average of the right element.

    Returns:
      The slope limited polynomial in the current element.
    """
    n_p, k = u_x.shape
    if x_l.shape[0] != n_p or x_l.shape[1] != k:
        raise ValueError('The shape of input polynomial and collocation '
                'points mismatches: u_l ({}), x_l({})'.format( \
                        u_x.shape, x_l.shape))
    # Computes the grid size.
    h = x_l[-1, :] - x_l[0, :]
    h_n = np.tile(h, (n_p, 1))

    # Computes the coordinates at the middle point of elements.
    x_0 = np.tile(x_l[0, :] + 0.5 * h, (n_p, 1))

    # Computes the limit function.
    return np.tile(v_0, (n_p, 1)) + (x_l - x_0) * ( \
            np.tile( \
            minmod(np.array( \
            [u_x[0, :], (v_p1 - v_0) / h, (v_0 - v_m1) / h])), \
            (n_p, 1)))
