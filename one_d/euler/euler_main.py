import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import matplotlib.pyplot as plt
import numpy as np

from one_d import equation
from one_d.euler import euler
from one_d import mesh_gen_1d

_DTYPE = np.float32

def main():
    """The driver script for solving the 1D Euler equations."""
    # Polynomial order ued for approximation.
    n = 6

    # Number of elements.
    k = 250

    # Generate the mesh.
    x_min = 0.0
    x_max = 1.0
    n_v, v_x, e_to_v = mesh_gen_1d.equidistant(x_min, x_max, k)

    # initialize the global context.
    config = equation.Equation(n, v_x, e_to_v)

    # Set up initial conditions: the Sod's problem.
    mass_matrix = np.matmul(np.linalg.inv(config.v.T), config.inv_v)
    c_x = np.tile(np.sum(np.matmul(mass_matrix, config.x), axis=0), \
            (config.n_p, 1))

    rho = np.where(c_x < 0.5, np.ones_like(c_x), 0.125 * np.ones_like(c_x))
    rhou = np.zeros((config.n_p, config.k), dtype=_DTYPE)
    e = np.where(c_x < 0.5, np.ones_like(c_x), 0.1 * np.ones_like(c_x)) / \
            (euler.GAMMA - 1.0)
    states = {'rho': rho, 'rhou': rhou, 'E': e}

    t_final = 0.2

    # Solve the problem.
    results = euler.euler_solve(config, states, t_final)

    # Show the results.
    x = np.reshape(config.x, (np.prod(config.x.shape),))
    e_0 = np.reshape(e, (np.prod(e.shape),))
    e_t = np.reshape(results['E'], (np.prod(results['E'].shape),))
    rhou_0 = np.reshape(rhou, (np.prod(rhou.shape),))
    rhou_t = np.reshape(results['rhou'], (np.prod(results['rhou'].shape),))
    rho_0 = np.reshape(rho, (np.prod(rho.shape),))
    rho_t = np.reshape(results['rho'], (np.prod(results['rho'].shape),))

    u_0 = rhou_0 / rho_0
    u_t = rhou_t / rho_t

    p_0 = euler.pressure(rho_0, rhou_0, e_0)
    p_t = euler.pressure(rho_t, rhou_t, e_t)

    c_0 = np.sqrt(euler.GAMMA * p_0 / rho_0)
    c_t = np.sqrt(euler.GAMMA * p_t / rho_t)

    m_0 = u_0 / c_0
    m_t = u_t / c_t

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(x, rho_0, 'o', x, rho_t, '*')
    ax[0, 1].plot(x, u_0, 'o', x, u_t, '*')
    ax[1, 0].plot(x, p_0, 'o', x, p_t, '*')
    ax[1, 1].plot(x, m_0, 'o', x, m_t, '*')
    plt.show()


if __name__ == '__main__':
    main()
