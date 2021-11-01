import numpy as np

import equation
import mesh_gen_1d

if __name__ == '__main__':
    n = 8
    n_v, v_x, e_to_v = mesh_gen_1d.equidistant(0.0, 2.0, 10)
    config = equation.Equation(n, v_x, e_to_v)
    u = np.sin(config.x)
    print('SLOPE_LIMIT_N = {}'.format(config.slope_limit_n(u)))
    print('LIFT = {}'.format(config.lift))
    print('F_SCALE = {}'.format(config.f_scale))
