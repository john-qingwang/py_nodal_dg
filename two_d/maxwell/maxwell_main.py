import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp_io

from two_d import equation
from two_d.maxwell import maxwell


def main():
    """The driver script for solving the 2D Maxwell equation in TM form."""
    # Polynomial order ued for approximation.
    n = 10

    # Read the mesh.
    mesh = sp_io.loadmat('../test_data/maxwell.mat')

    # Initialize solver and construct grid and metric.
    config = equation.Equation( \
            n,
            np.squeeze(mesh['VX']),
            np.squeeze(mesh['VY']),
            mesh['EToV'])

    # Set initial conditions.
    m_mode = 1
    n_mode = 1
    states = {
        'Hx': np.zeros((config.n_p, config.k), dtype=equation.DTYPE),
        'Hy': np.zeros((config.n_p, config.k), dtype=equation.DTYPE),
        'Ez':
            np.sin(m_mode * np.pi * config.x) * \
            np.sin(n_mode * np.pi * config.y),
    }

    # Solve the problem.
    t_final = 1.0
    states_new = maxwell.maxwell_2d_solve(config, states, t_final)

    # Show the final results.
    fig, ax = plt.subplots(1, 3)
    ax[0].tricontour(
            config.flatten(config.x),
            config.flatten(config.y),
            config.flatten(states_new['Hx']))
    ax[1].tricontour(
            config.flatten(config.x),
            config.flatten(config.y),
            config.flatten(states_new['Hy']))
    ax[2].tricontour(
            config.flatten(config.x),
            config.flatten(config.y),
            config.flatten(states_new['Ez']))
    plt.show()


if __name__ == '__main__':
    main()
