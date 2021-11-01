import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import numpy as np

from one_d import time_lib

GAMMA = 1.4


def _rho_f(rhou):
    """Computes the flux of the continuity equation."""
    return rhou


def _rhou_f(rho, rhou, p):
    """Computes the flux of the momentum equation."""
    return rhou**2 / rho + p


def _e_f(rho, rhou, e, p):
    """Computes the flux of the energy equation."""
    return (e + p) * rhou / rho


def pressure(rho, rhou, e):
    """Computes the pressure."""
    return (GAMMA - 1.0) * (e - 0.5 * rhou**2 / rho)


def _lax_friedrichs_flux(d_f, d_u, n_x, c):
    """Computes the Lax-Friedrichs flux."""
    return 0.5 * n_x * d_f + 0.5 * c * d_u


def euler_rhs(config, states, time_local, dt):
    """Evaluates the RHS flux in 1D Euler equation."""
    del time_local, dt

    rho = states['rho']
    rhou = states['rhou']
    e = states['E']
    u = rhou / rho

    # Compute the maximum velocity for Lax-Friedrichs flux.
    p = pressure(rho, rhou, e)
    c = np.sqrt(GAMMA * p / rho)
    lm = np.abs(u) + c

    # Compute fluxes.
    rho_f = _rho_f(rhou)
    rhou_f = _rhou_f(rho, rhou, p)
    e_f = _e_f(rho, rhou, e, p)

    # Compute jumps at internal faces.
    rho_buf = config.flatten(rho)
    rhou_buf = config.flatten(rhou)
    e_buf = config.flatten(e)
    rho_f_buf = config.flatten(rho_f)
    rhou_f_buf = config.flatten(rhou_f)
    e_f_buf = config.flatten(e_f)
    lm_buf = config.flatten(lm)

    d_rho = rho_buf[config.v_map_m] - rho_buf[config.v_map_p]
    d_rhou = rhou_buf[config.v_map_m] - rhou_buf[config.v_map_p]
    d_e = e_buf[config.v_map_m] - e_buf[config.v_map_p]
    d_rho_f = rho_f_buf[config.v_map_m] - rho_f_buf[config.v_map_p]
    d_rhou_f = rhou_f_buf[config.v_map_m] - rhou_f_buf[config.v_map_p]
    d_e_f = e_f_buf[config.v_map_m] - e_f_buf[config.v_map_p]

    lf_c = np.maximum(lm_buf[config.v_map_m], lm_buf[config.v_map_p])

    # Compute Lax-Friedrichs fluxes at interfaces.
    n_x = config.flatten(config.n_x)

    d_rho_f = _lax_friedrichs_flux(d_rho_f, d_rho, n_x, -lf_c)
    d_rhou_f = _lax_friedrichs_flux(d_rhou_f, d_rhou, n_x, -lf_c)
    d_e_f = _lax_friedrichs_flux(d_e_f, d_e, n_x, -lf_c)

    # Apply boundary conditions for the Sod's problem.
    rho_i = 1.0
    rhou_i = 0.0
    p_i = 1.0
    e_i = p_i / (GAMMA - 1.0)

    rho_o = 0.125
    rhou_o = 0.0
    p_o = 0.1
    e_o = p_o / (GAMMA - 1.0)

    # Set fluxes at inflow.
    rho_f_i = _rho_f(rhou_i)
    rhou_f_i = _rhou_f(rho_i, rhou_i, p_i)
    e_f_i = _e_f(rho_i, rhou_i, e_i, p_i)
    lm_i = lm_buf[config.v_map_i]
    n_x_i = n_x[config.map_i]

    d_rho_f[config.map_i] = _lax_friedrichs_flux( \
            rho_f_buf[config.v_map_i] - rho_f_i, \
            rho_buf[config.v_map_i] - rho_i, \
            n_x_i, -lm_i)
    d_rhou_f[config.map_i] = _lax_friedrichs_flux( \
            rhou_f_buf[config.v_map_i] - rhou_f_i, \
            rhou_buf[config.v_map_i] - rhou_i, \
            n_x_i, -lm_i)
    d_e_f[config.map_i] = _lax_friedrichs_flux( \
            e_f_buf[config.v_map_i] - e_f_i, \
            e_buf[config.v_map_i] - e_i, \
            n_x_i, -lm_i)

    # Set fluxes at outflow.
    rho_f_o = _rho_f(rhou_o)
    rhou_f_o = _rhou_f(rho_o, rhou_o, p_o)
    e_f_o = _e_f(rho_o, rhou_o, e_o, p_o)
    lm_o = lm_buf[config.v_map_o]
    n_x_o = n_x[config.map_o]

    d_rho_f[config.map_o] = _lax_friedrichs_flux( \
            rho_f_buf[config.v_map_o] - rho_f_o, \
            rho_buf[config.v_map_o] - rho_o, \
            n_x_o, -lm_o)
    d_rhou_f[config.map_o] = _lax_friedrichs_flux( \
            rhou_f_buf[config.v_map_o] - rhou_f_o, \
            rhou_buf[config.v_map_o] - rhou_o, \
            n_x_o, -lm_o)
    d_e_f[config.map_o] = _lax_friedrichs_flux( \
            e_f_buf[config.v_map_o] - e_f_o, \
            e_buf[config.v_map_o] - e_o, \
            n_x_o, -lm_o)

    # Compute right hand sides of the Euler equations.
    n_0 = config.n_fp * config.n_faces
    n_1 = config.k
    d_rho_f = config.expand(d_rho_f, n_0, n_1)
    d_rhou_f = config.expand(d_rhou_f, n_0, n_1)
    d_e_f = config.expand(d_e_f, n_0, n_1)

    def rhs_fn(f, df):
        """Computs the right hand side function."""
        return -config.rx * np.matmul(config.dr, f) + \
                np.matmul(config.lift, config.f_scale * df)

    return {'rho': rhs_fn(rho_f, d_rho_f), \
            'rhou': rhs_fn(rhou_f, d_rhou_f), \
            'E': rhs_fn(e_f, d_e_f)}
    

def euler_solve(config, states, t_final):
    """Solves the 1D Euler equation."""
    # Parameters.
    cfl = 1.0
    time = 0.0

    # Prepare for adaptive time stepping.
    dx_min = np.min(config.x[1, :] - config.x[0, :])

    # Limit initial solution.
    states = {k: config.slope_limit_n(v) for k, v in states.items()}

    # Outer time step loop.
    while time < t_final:
        u = states['rhou'] / states['rho']
        temp = states['E'] / states['rho'] - 0.5 * u**2
        cvel = np.sqrt(GAMMA  * (GAMMA - 1) * temp)
        dt = cfl * np.min(dx_min / (np.abs(u) + cvel))

        if time + dt > t_final:
            dt = t_final - time

        time, states = time_lib.rk3(euler_rhs, states, config, time, dt)
        print('t = {}'.format(time))

    return states
