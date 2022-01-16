"""A library of utility functions for the 2D Euler equation."""
import numpy as np

from two_d import numerics_2d

def fluxes_and_primitive_variables(q, gamma):
    """Evaluates Euler fluxes and primitive variables."""
    # Extract conserved variables.
    rho = q[..., 0]
    rhou = q[..., 1]
    rhov = q[..., 2]
    e = q[..., 3]

    # Compute primitive variables.
    u = rhou / rho
    v = rhov / rho
    p = (gamma - 1) * (e - 0.5 * (rhou * u + rhov * v))

    # Compute flux functions.
    f = np.zeros_like(q)
    f[..., 0] = rhou
    f[..., 1] = rhou * u + p
    f[..., 2] = rhov * u
    f[..., 3] = u * (e + p)

    g = np.zeros_like(q)
    g[..., 0] = rhov
    g[..., 1] = rhou * v
    g[..., 2] = rhov * v + p
    g[..., 3] = v * (e + p)

    return f, g, rho, u, v, p


def flux_hll(n_x, n_y, q_m, q_p, gamma):
    """Computes Harten-Lax-van Leer surface fluxes."""
    # Rotate "-" trace momentum to face normal-tangent coordinates.
    rhou_m = np.copy(q_m[..., 1])
    rhov_m = np.copy(q_m[..., 2])
    e_m = q_m[..., 3]
    q_m[..., 1] = n_x * rhou_m + n_y * rhov_m
    q_m[..., 2] = -n_y * rhou_m + n_x * rhov_m

    # Rotate "+" trace momentum to face normal-tangent coordinates.
    rhou_p = np.copy(q_p[..., 1])
    rhov_p = np.copy(q_p[..., 2])
    e_p = q_p[..., 3]
    q_p[..., 1] = n_x * rhou_p + n_y * rhov_p
    q_p[..., 2] = -n_y * rhou_p + n_x * rhov_p

    # Compute fluxes and primitive variables in rotated coordinates.
    f_m, _, rho_m, u_m, v_m, p_m = fluxes_and_primitive_variables(q_m, gamma)
    f_p, _, rho_p, u_p, v_p, p_p = fluxes_and_primitive_variables(q_p, gamma)

    def enthalpy(e, p, rho):
        """Computes the mass specific enthalpy."""
        return (e + p) / rho

    def sound_speed(p, rho):
        """Computes the speed of sound."""
        return np.sqrt(gamma * p / rho)

    h_m = enthalpy(e_m, p_m, rho_m)
    c_m = sound_speed(p_m, rho_m)
    h_p = enthalpy(e_p, p_p, rho_p)
    c_p = sound_speed(p_p, rho_p)

    # Compute the Roe average variables.
    rho_ms = np.sqrt(rho_m)
    rho_ps = np.sqrt(rho_p)

    rho = rho_ms * rho_ps
    u = (rho_ms * u_m + rho_ps * u_p) / (rho_ms + rho_ps)
    v = (rho_ms * v_m + rho_ps * v_p) / (rho_ms + rho_ps)
    h = (rho_ms * h_m + rho_ps * h_p) / (rho_ms + rho_ps)

    c = np.sqrt((gamma - 1) * (h - 0.5 * (u**2 + v**2)))

    # Compute estimate of waves speeds.
    s_l = np.minimum(u_m - c_m, u - c)
    s_r = np.maximum(u_p + c_p, u + c)

    # Compute the HLL flux.
    t1 = numerics_2d.divide_no_nan( \
            np.minimum(s_r, 0) - np.minimum(s_l, 0), s_r - s_l)
    t2 = 1.0 - t1
    t3 = numerics_2d.divide_no_nan( \
            s_r * np.abs(s_l) - s_l * np.abs(s_r), 2.0 * (s_r - s_l))

    f_x = t1[..., np.newaxis] * f_p + t2[..., np.newaxis] * f_m - \
        t3[..., np.newaxis] * (q_p - q_m)

    # Rotate flux back into Cartesian coordinates.
    flux = np.copy(f_x)
    flux[..., 1] = n_x * f_x[..., 1] - n_y * f_x[..., 2]
    flux[..., 2] = n_y * f_x[..., 1] + n_x * f_x[..., 2]

    return flux


def limiter(config, q, bc, time):
    """Limits the slope of the Euler solution."""
    # Gas constant.
    gamma = 1.4

    # 1. Compute geometric information for 4 element patch ocntaining each
    # element.
    # Build the average matrix.
    ave = 0.5 * np.sum(config.m, axis=0)

    # Compute displacements from center of nodes for Taylor expansion of limited
    # fields.
    drop_ave = np.eye(config.n_p) - np.tile(ave, (config.n_p, 1))
    dx = np.matmul(drop_ave, config.x)
    dy = np.matmul(drop_ave, config.y)

    # Find neighbors in patch.
    e1 = config.e_to_e[:, 0]
    e2 = config.e_to_e[:, 1]
    e3 = config.e_to_e[:, 2]

    # Extract coordinates of vertices and centers of elements.
    v1 = self.e_to_v[:, 0]
    x_v1 = config.v_x[v1]
    y_v1 = config.v_y[v1]
    v2 = self.e_to_v[:, 1]
    x_v2 = config.v_x[v2]
    y_v2 = config.v_y[v2]
    v3 = self.e_to_v[:, 2]
    x_v3 = config.v_x[v3]
    y_v3 = config.v_y[v3]

    # Compute face unit normals and lengths.
