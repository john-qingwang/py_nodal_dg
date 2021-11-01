import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import numpy as np

from one_d import equation
from one_d import time_lib


def maxwell_1d_rhs(config, states, t, eps, mu):
    """Evaluates the RHS in 1D Maxwell equations."""
    del t

    z_imp = config.flatten(np.sqrt(mu / eps))
    e = config.flatten(states['E'])
    h = config.flatten(states['H'])

    # Define field difference at faces.
    de = e[config.v_map_m] - e[config.v_map_p]
    dh = h[config.v_map_m] - h[config.v_map_p]
    z_imp_m = config.expand(z_imp[config.v_map_m], \
            config.n_fp * config.n_faces, config.k)
    z_imp_p = config.expand(z_imp[config.v_map_p], \
            config.n_fp * config.n_faces, config.k)
    y_imp_m = 1.0 / z_imp_m
    y_imp_p = 1.0 / z_imp_p

    # Apply homogeneous boundary conditions, Ez = 0.
    e_bc = -e[config.v_map_b]
    h_bc = h[config.v_map_b]
    de[config.map_b] = e[config.v_map_b] - e_bc
    dh[config.map_b] = h[config.v_map_b] - h_bc
    de = config.expand(de, config.n_fp * config.n_faces, config.k)
    dh = config.expand(dh, config.n_fp * config.n_faces, config.k)

    # Evaluate upwind fluxes.
    flux_e = 1.0 / (z_imp_m + z_imp_p) * (config.n_x * z_imp_p * dh - de)
    flux_h = 1.0 / (y_imp_m + y_imp_p) * (config.n_x * y_imp_p * de - dh)

    # Compute right hand sides of the PDE's.
    rhs_e = (-config.rx * np.matmul(config.dr, states['H']) + \
            np.matmul(config.lift, config.f_scale * flux_e)) / eps
    rhs_h = (-config.rx * np.matmul(config.dr, states['E']) + \
            np.matmul(config.lift, config.f_scale * flux_h)) / mu
    
    return {'E': rhs_e, 'H': rhs_h}


def maxwell_1d_solve(config, states, t_final, eps, mu):
    """Solves the 1D Maxwell eqaution until `t_final`."""
    t = 0

    # Compute the time step size.
    x_min = np.min(np.abs(config.x[0, :] - config.x[1, :]))
    cfl = 1.0
    dt = cfl * x_min
    n_step = int(np.ceil(t_final / dt))
    dt = t_final / n_step

    # Integrate in time.
    for t_step in range(n_step):
        t, states = time_lib.rk4(
                maxwell_1d_rhs, states, config, t, dt, eps=eps, mu=mu)

    return states


