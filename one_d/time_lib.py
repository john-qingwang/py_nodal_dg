import numpy as np

_RK4A = [0.0, \
        -567301805773.0 / 1357537059087.0, \
        -2404267990393.0 / 2016746695238.0, \
        -3550918686646.0 /2091501179385.0, \
        -1275806237668.0 / 842570457699.0];
_RK4B = [1432997174477.0 / 9575080441755.0, \
        5161836677717.0 / 13612068292357.0, \
        1720146321549.0 / 2090206949498.0, \
        3134564353537.0 / 4481467310338.0, \
        2277821191437.0 / 14882151754819.0];
_RK4C = [0.0, \
        1432997174477.0 / 9575080441755.0, \
        2526269341429.0 / 6820363962896.0, \
        2006345519317.0 / 3224310063776.0, \
        2802321613138.0 / 2924317926251.0];


def rk4(rhs, states, config, time, dt, **kwargs):
    """Integrates an ODE du/dt = rhs(config, states, time, **kwargs)."""
    res_u = {key: np.zeros((config.n_p, config.k)) \
            for key in states.keys()}

    for i in range(5):
        time_local = time + _RK4C[i] * dt
        rhs_u = rhs(config, states, time_local, **kwargs)
        res_u = {key: _RK4A[i] * res_u[key] + dt * rhs_u[key] \
                for key in states.keys()}
        states = {key: val + _RK4B[i] * res_u[key] \
                for key, val in states.items()}
        
    return time + dt, states


def rk3(rhs, states, config, time, dt, **kwargs):
    """3rd order SSP RK for du/dt = rhs(config, states, time, **kwargs)."""
    # SSP RK stage 1.
    rhs_val = rhs(config, states, time, dt, **kwargs)
    states_1 = {k: v + dt * rhs_val[k] for k, v in states.items()}
    states_1 = {k: config.slope_limit_n(v) for k, v in states_1.items()}

    # SSP RK stage 2.
    rhs_val = rhs(config, states_1, time, dt, **kwargs)
    states_2 = {k: (3.0 * v + states_1[k] + dt * rhs_val[k]) / 4.0 \
            for k, v in states.items()}
    states_2 = {k: config.slope_limit_n(v) for k, v in states_2.items()}

    # SSP RK Stage 3.
    rhs_val = rhs(config, states_2, time, dt, **kwargs)
    states = {k: (v + 2.0 * states_2[k] + 2.0 * dt * rhs_val[k]) / 3.0 \
            for k, v in states.items()}
    states = {k: config.slope_limit_n(v) for k, v in states.items()}
        
    return time + dt, states
