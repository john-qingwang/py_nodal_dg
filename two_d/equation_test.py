import sys
sys.path.insert(1, '/Users/qingwang/Documents/research/nodal-dg')

import numpy as np

from two_d import equation


def test_f_mask():
    """Checks if the face mask is implemented correctly."""
    r = np.array([0.0, 0.3, -1.0, 0.2, -0.6, -1.0, 0.5])
    s = np.array([-1.0, -0.3, -1.0, 0.7, 0.6, 0.0, 0.5])
    f_mask = equation.Equation.get_face_mask(r, s)

    expected = np.array([[0, 2], [1, 4], [2, 5]]).transpose()
    assert(np.all(expected == f_mask))


def test():
    """Tests functions in the 2D equation library."""
    test_f_mask()


if __name__ == '__main__':
    test()
