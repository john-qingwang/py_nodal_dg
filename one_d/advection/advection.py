import numpy as np

from one_d import equation
from one_d import time_lib


def advection_rhs(config, states, t, a):
    """Computes the rhs of the advection equation."""
    u = states['u']
    u_buf = np.reshape(u, (np.prod(u.shape),))

    # Form field differences at faces.
    alpha = 1.0
    n_x = np.reshape(config.n_x, (np.prod(config.n_x.shape),))
    du = 0.5 * (u_buf[config.v_map_m] - u_buf[config.v_map_p]) * \
            (a * n_x - (1.0 - alpha) * np.abs(a * n_x))

    # Impose boundary conditions at x = 0.
    u_in = -np.sin(a * t)
    du[config.map_i] = 0.5 * (u_buf[config.map_i] - u_in) * \
            (a * n_x[config.map_i] - \
            (1.0 - alpha) * np.abs(a * n_x[config.map_i]))
    du[config.map_o] = 0.0

    du = np.reshape(du, (config.n_fp * config.n_faces, config.k))

    # Compute right hand sides of the semi-discrete PDE.
    return {'u': -a * config.rx * np.matmul(config.dr, u) + \
            np.matmul(config.lift, config.f_scale * du)}


def advection_solve(config, states, a, t_final):
    """Solves the 1D advection equation."""
    t = 0.0

    # Compute time step size.
    x_min = np.min(np.abs(config.x[0, :] - config.x[1, :]))
    cfl = 0.75
    dt = 0.5 * cfl / a * x_min
    n_step = int(np.ceil(t_final / dt))
    dt = t_final / n_step

    # Integrate in time.
    for t_step in range(n_step):
        t, states = time_lib.rk4(
                advection_rhs, states, config, t, dt, a=a)

    return states
