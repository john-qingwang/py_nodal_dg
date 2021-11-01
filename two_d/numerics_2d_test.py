import numpy as np

import numerics_2d


def test():
    """Checks results from numerics_2d."""
    n = 2

    x, y = numerics_2d.nodes_2d(n)

    r, s = numerics_2d.xy_to_rs(x, y)

    a, b = numerics_2d.rs_to_ab(r, s)

    dmdr, dmds = numerics_2d.grad_simplex_2d_p(a, b, 0, 1)

    vr, vs = numerics_2d.grad_vandermonde_2d(n, r, s)

    print('V_r = {}'.format(vr))
    print('V_s = {}'.format(vs))

    v = numerics_2d.vandermonde_2d(n, r, s)

    print('V = {}'.format(v))

    dr, ds = numerics_2d.d_matrices_2d(n, r, s, v)

    print('Dr = {}'.format(dr))
    print('Ds = {}'.format(ds))

if __name__ == '__main__':
    test()
