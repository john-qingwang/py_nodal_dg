import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import numpy as np

from two_d import equation
from one_d import time_lib

ALPHA = 1.0


def maxwell_2d_rhs(config, states, t):
    """Evaluates the RHS in the 2D Maxwell equation in TM form."""
    hx = config.flatten(states['Hx'])
    hy = config.flatten(states['Hy'])
    ez = config.flatten(states['Ez'])

    # Define field differences at faces.
    d_hx = hx[config.v_map_m] - hx[config.v_map_p]
    d_hy = hy[config.v_map_m] - hy[config.v_map_p]
    d_ez = ez[config.v_map_m] - ez[config.v_map_p]

    # Impose reflective boundary conditions (Ez+ = -Ez-).
    d_hx[config.map_b] = 0.0
    d_hy[config.map_b] = 0.0
    d_ez[config.map_b] = 2.0 * ez[config.v_map_b]

    d_hx = config.expand(d_hx, config.n_fp * config.n_faces, config.k)
    d_hy = config.expand(d_hy, config.n_fp * config.n_faces, config.k)
    d_ez = config.expand(d_ez, config.n_fp * config.n_faces, config.k)

    # Evaluate upwind fluxes.
    n_dot_dh = config.n_x * d_hx + config.n_y * d_hy
    flux_hx = config.n_y * d_ez + ALPHA * (n_dot_dh * config.n_x - d_hx)
    flux_hy = -config.n_x * d_ez + ALPHA * (n_dot_dh * config.n_y - d_hy)
    flux_ez = -config.n_x * d_hy + config.n_y * d_hx - ALPHA * d_ez

    # Compute local derivatives of fields.
    ez_x, ez_y = config.grad(states['Ez'])
    _, _, curl_hz = config.curl(states['Hx'], states['Hy'])

    # Compute right hand size of the PDEs.
    return {
        'Hx': -ez_y + np.matmul(config.lift, 0.5 * config.f_scale * flux_hx),
        'Hy': ez_x + np.matmul(config.lift, 0.5 * config.f_scale * flux_hy),
        'Ez': curl_hz + np.matmul(config.lift, 0.5 * config.f_scale * flux_ez),
    }


def maxwell_2d_solve(config, states, t_final):
    """Solves the 2D Maxwell equation till `t_final`."""
    time = 0.0

    while time < t_final:
        time, states = time_lib.rk4(
                maxwell_2d_rhs, states, config, time, config.dt)

    return states
