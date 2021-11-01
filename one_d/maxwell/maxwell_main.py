import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import matplotlib.pyplot as plt
import numpy as np

from one_d.maxwell import maxwell
from one_d import equation
from one_d import mesh_gen_1d

_DTYPE = np.float32


def main():
    """The main function for the advection equation."""
    # The order of polynomials used for approximation.
    n = 6

    # Generates a simple mesh.
    x_min = -1.0
    x_max = 1.0
    k = 80
    n_v, v_x, e_to_v = mesh_gen_1d.equidistant(x_min, x_max, k)

    # Initialize the solver and construct grid and metric.
    config = equation.Equation(n, v_x, e_to_v)

    # Set up material parameters.
    eps_1 = np.ones((k,), dtype=_DTYPE)
    eps_1[k // 2:] = 2.0
    eps = np.array([eps_1,] * config.n_p)
    mu = np.array([np.ones((k,), dtype=_DTYPE),] * config.n_p)

    # Set initial conditions.
    e = np.where(config.x < 0, np.sin(np.pi * config.x), \
            np.zeros_like(config.x))
    h = np.zeros((config.n_p, config.k), dtype=_DTYPE)

    # Solve the problem.
    t_final = 10.0
    results = maxwell.maxwell_1d_solve(
            config, {'E': np.copy(e), 'H': np.copy(h)}, t_final, \
                    eps=eps, mu=mu)

    # Show the result.
    x = np.reshape(config.x, (np.prod(config.x.shape),))
    e_0 = np.reshape(e, (np.prod(e.shape),))
    e_t = np.reshape(results['E'], (np.prod(results['E'].shape),))
    h_0 = np.reshape(h, (np.prod(h.shape),))
    h_t = np.reshape(results['H'], (np.prod(results['H'].shape),))
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, e_0, 'o', x, e_t, '*')
    ax[1].plot(x, h_0, 'o', x, h_t, '*')
    plt.show()


if __name__ == '__main__':
    main()
