"""A library for 1D mesh generation."""
import numpy as np


def equidistant(x_min, x_max, k):
    """Generates simple equidistant grid with `k` elements."""
    n_v = k + 1

    # Generate node coordinates.
    dx = (x_max - x_min) / k
    v_x = np.array([i * dx + x_min for i in range(n_v)])

    # Read element to node connectivity.
    e_to_v = np.array([[i, i + 1] for i in range(k)])

    return n_v, v_x, e_to_v
