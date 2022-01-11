import matplotlib.pyplot as plt
import numpy as np

from one_d.advection import advection
from one_d import equation
from one_d import mesh_gen_1d

def main():
    """The main function for the advection equation."""
    # The order of polynomials used for approximation.
    n = 8

    # Generates a simple mesh.
    x_min = 0.0
    x_max = 2.0
    k = 10
    n_v, v_x, e_to_v = mesh_gen_1d.equidistant(x_min, x_max, k)

    # Initialize the solver and construct grid and metric.
    config = equation.Equation(n, v_x, e_to_v)

    # Advection speed.
    a = 2.0 * np.pi

    # Set initial conditions.
    u = np.sin(config.x)

    # Solve the problem.
    time_final = 10.0
    results = advection.advection_solve(\
            config, {'u': np.copy(u)}, a, time_final)

    # Show the result.
    x = np.reshape(config.x, (np.prod(config.x.shape),))
    u_0 = np.reshape(u, (np.prod(u.shape),))
    u_t = np.reshape(results['u'], (np.prod(results['u'].shape),))
    fig, ax = plt.subplots()
    ax.plot(x, u_0, 'o', x, u_t, '*')
    plt.show()


if __name__ == '__main__':
    main()
