"""A library of utility functions for the 2D Euler equation."""
import enum
import numpy as np

from two_d import boudnary_condition, numerics_2d

BoundaryCondition = boudnary_condition.BoundaryCondition


def to_primitive(rho, rhou, rhov, ener, gamma=1.4):
    """Converts conservative variables to primitive variables rho, u, v, and p."""
    u = rhou / rho
    v = rhov / rho
    p = (gamma - 1.0) * (ener - 0.5 * rho * (u**2 + v**2))
    return rho, u, v, p


def fluxes_and_primitive_variables(q, gamma):
    """Evaluates Euler fluxes and primitive variables."""
    # Extract conserved variables.
    rho = q[..., 0]
    rhou = q[..., 1]
    rhov = q[..., 2]
    e = q[..., 3]

    # Compute primitive variables.
    rho, u, v, p = to_primitive(rho, rhou, rhov, e, gamma)

    # Compute flux functions.
    f = np.zeros_like(q)
    f[..., 0] = rhou
    f[..., 1] = rhou * u + p
    f[..., 2] = rhov * u
    f[..., 3] = u * (e + p)

    g = np.zeros_like(q)
    g[..., 0] = rhov
    g[..., 1] = rhou * v
    g[..., 2] = rhov * v + p
    g[..., 3] = v * (e + p)

    return f, g, rho, u, v, p


def flux_hll(n_x, n_y, q_m, q_p, gamma):
    """Computes Harten-Lax-van Leer surface fluxes."""
    # Rotate "-" trace momentum to face normal-tangent coordinates.
    rhou_m = np.copy(q_m[..., 1])
    rhov_m = np.copy(q_m[..., 2])
    e_m = q_m[..., 3]
    q_m[..., 1] = n_x * rhou_m + n_y * rhov_m
    q_m[..., 2] = -n_y * rhou_m + n_x * rhov_m

    # Rotate "+" trace momentum to face normal-tangent coordinates.
    rhou_p = np.copy(q_p[..., 1])
    rhov_p = np.copy(q_p[..., 2])
    e_p = q_p[..., 3]
    q_p[..., 1] = n_x * rhou_p + n_y * rhov_p
    q_p[..., 2] = -n_y * rhou_p + n_x * rhov_p

    # Compute fluxes and primitive variables in rotated coordinates.
    f_m, _, rho_m, u_m, v_m, p_m = fluxes_and_primitive_variables(q_m, gamma)
    f_p, _, rho_p, u_p, v_p, p_p = fluxes_and_primitive_variables(q_p, gamma)

    def enthalpy(e, p, rho):
        """Computes the mass specific enthalpy."""
        return (e + p) / rho

    def sound_speed(p, rho):
        """Computes the speed of sound."""
        return np.sqrt(gamma * p / rho)

    h_m = enthalpy(e_m, p_m, rho_m)
    c_m = sound_speed(p_m, rho_m)
    h_p = enthalpy(e_p, p_p, rho_p)
    c_p = sound_speed(p_p, rho_p)

    # Compute the Roe average variables.
    rho_ms = np.sqrt(rho_m)
    rho_ps = np.sqrt(rho_p)

    rho = rho_ms * rho_ps
    u = (rho_ms * u_m + rho_ps * u_p) / (rho_ms + rho_ps)
    v = (rho_ms * v_m + rho_ps * v_p) / (rho_ms + rho_ps)
    h = (rho_ms * h_m + rho_ps * h_p) / (rho_ms + rho_ps)

    c = np.sqrt((gamma - 1) * (h - 0.5 * (u**2 + v**2)))

    # Compute estimate of waves speeds.
    s_l = np.minimum(u_m - c_m, u - c)
    s_r = np.maximum(u_p + c_p, u + c)

    # Compute the HLL flux.
    t1 = numerics_2d.divide_no_nan(
        np.minimum(s_r, 0) - np.minimum(s_l, 0), s_r - s_l)
    t2 = 1.0 - t1
    t3 = numerics_2d.divide_no_nan(
        s_r * np.abs(s_l) - s_l * np.abs(s_r), 2.0 * (s_r - s_l))

    f_x = t1[..., np.newaxis] * f_p + t2[..., np.newaxis] * f_m - \
        t3[..., np.newaxis] * (q_p - q_m)

    # Rotate flux back into Cartesian coordinates.
    flux = np.copy(f_x)
    flux[..., 1] = n_x * f_x[..., 1] - n_y * f_x[..., 2]
    flux[..., 2] = n_y * f_x[..., 1] + n_x * f_x[..., 2]

    return flux


def limiter(config, q, bc_fn, time):
    """Limits the slope of the Euler solution."""
    # Gas constant.
    gamma = 1.4

    # 1. Compute geometric information for 4 element patch ocntaining each
    # element.
    # Build the average matrix.
    ave = 0.5 * np.sum(config.m, axis=0)

    # Compute displacements from center of nodes for Taylor expansion of limited
    # fields.
    drop_ave = np.eye(config.n_p) - np.tile(ave, (config.n_p, 1))
    dx = np.matmul(drop_ave, config.x)
    dy = np.matmul(drop_ave, config.y)

    # Find neighbors in patch.
    e1, e2, e3 = [config.e_to_e[:, i] for i in range(3)]

    # Extract coordinates of vertices and centers of elements.
    v1, v2, v3 = [config.e_to_v[:, i] for i in range(3)]
    x_v1, x_v2, x_v3 = [config.v_x[v] for v in range((v1, v2, v3))]
    y_v1, y_v2, y_v3 = [config.v_y[v] for v in range((v1, v2, v3))]

    # Compute face unit normals and lengths.
    fn_x = np.array([y_v2 - y_v1, y_v3 - y_v2, y_v1 - y_v3])
    fn_y = np.array([x_v2 - x_v1, x_v3 - x_v2, x_v1 - x_v3])
    fl = np.sqrt(fn_x**2 + fn_y**2)
    fn_x /= fl
    fn_y /= fl

    # Compute element centers.
    xc_0 = np.matmul(ave, config.x)
    xc = [xc_0[e] for e in (e1, e2, e3)]
    yc_0 = np.matmul(ave, config.y)
    yc = [yc_0[e] for e in (e1, e2, e3)]

    # Compute weights for face gradients.
    a0 = np.matmul(ave, config.jac * 2.0 / 3.0)
    a = [a0 + a0[e] for e in (e1, e2, e3)]

    # Find boundary faces for each face.
    id = [np.where(config.bc_type[:, i] > 0) for i in range(3)]

    # Compute location of centers of relfected ghost elements at boundary faces.
    h = [2.0 * a0[id_i] / fl[i, id_i] for i, id_i in enumerate(id)]
    for i, h_i in enumerate(h):
        xc[i][id[i]] += 2.0 * fn_x[i, id[i]] * h_i
        yc[i][id[i]] += 2.0 * fn_y[i, id[i]] * h_i

    # 2. Find cell averages of convserved & primitive variables in each 4 element patch.
    # Extract fields from Q.
    rho, rhou, rhov, ener = [q[..., i] for i in range(4)]

    # Compute cell averages of conserved variables.
    rho_c, rhou_c, rhov_c, ener_c = [
        np.matmul(ave, var) for var in (rho, rhou, rhov, ener)]
    ave_rho, ave_rhou, ave_rhov, ave_ener = [
        np.tile(var, (config.n_p, 1)) for var in (rho_c, rhou_c, rhov_c, ener_c)]

    # Compute primitive variables from cell averages of conserved variables.
    k = len(rho_c)
    pc_0 = np.zeros((1, k, 4), dtype=rho_c.dtype)
    pc_0[0, :, 0], pc_0[0, :, 1], pc_0[0, :, 2], pc_0[0, :,
                                                      3] = to_primitive(rho_c, rhou_c, rhov_c, ener_c, gamma)

    # Find neighbor values of convserved variables.
    pc = np.zeros((3, k, 4), dtype=rho_c.ctype)
    pc[..., 0] = rho_c[config.e_to_e.T]
    pc[..., 1] = rhou_c[config.e_to_e.T]
    pc[..., 2] = rhov_c[config.e_to_e.T]
    pc[..., 3] = ener_c[config.e_to_e.T]

    # Find boundary faces.
    id_w, id_i, id_o, id_c = [np.where(config.bc_type.T == bc_t for bc_t in (
        BoundaryCondition.WALL, BoundaryCondition.IN, BoundaryCondition.OUT, BoundaryCondition.CYL))]

    # Apply boundary conditions to cell averages of ghost cells at boundary faces.
    pc = bc_fn(xc, yc, fn_x, fn_y, id_i, id_o, id_w, id_c, pc, time)
    pc[..., 0], pc[..., 1], pc[..., 2], pc[..., 3] = to_primitive(
        pc[..., 0], pc[..., 1], pc[..., 2], pc[..., 3], gamma)

    # 3. Compute average of primitive variables at face nodes.
    ids = [1, config.n_fp, config.n_fp + 1, 2 *
           config.n_fp, 3 * config.n_fp, 2 * config.n_fp + 1]
    v_map_p1 = np.reshape(
        config.v_map_p, (config.n_fp * config.n_faces, config.k))[ids, :]
    v_map_m1 = np.reshape(
        config.v_map_m, (config.n_fp * config.n_faces, config.k))[ids, :]

    rho_a, rhou_a, rhov_a, ener_a = [
        0.5 * (var[v_map_p1] + var[v_map_m1]) for var in (rho, rhou, rhov, ener)]

    pv_a = np.zeros((len(ids), config.k, 4), dtype=rho_a.dtype)
    pv_a[..., 0], pv_a[..., 1], pv_a[..., 2], pv_a[...,
                                                   3] = to_primitive(rho_a, rhou_a, rhov_a, ener_a, gamma)

    # 4. Apply limiting procedure to each of the primitive variables.
    # Storage for cell averages and limited gradients at each node of each element.
    a_v = np.zeros((config.n_p, config.k, 4), dtype=q.dtype)
    d_v = np.zeros((config.n_p, config.k, 4), dtype=q.dtype)

    # Loop over primitve variables.
    eps = 1e-10
    for n in range(4):
        # Find value of primitive variables in patches.
        vc_0 = pc_0[0, :, n]
        vc_1, vc_2, vc_3 = [pc[i, :, n] for i in range(3)]
        va = pv_a[..., n]

        # Compute face gradients.
        dvdx_e1 = 0.5 * ((vc_1 - vc_0) * (y_v2 - y_v1) +
                         (va[0, :] - va[1, :]) * (yc[0] - yc_0)) / a[0]
        dvdy_e1 = -0.5 * ((vc_1 - vc_0) * (x_v2 - x_v1) +
                          (va[0, :] - va[1, :]) * (xc[0] - xc_0)) / a[0]
        dvdx_e2 = 0.5 * ((vc_2 - vc_0) * (y_v3 - y_v2) +
                         (va[2, :] - va[3, :]) * (yc[1] - yc_0)) / a[1]
        dvdy_e2 = -0.5 * ((vc_2 - vc_0) * (x_v3 - x_v2) +
                          (va[2, :] - va[3, :]) * (xc[1] - xc_0)) / a[1]
        dvdx_e3 = 0.5 * ((vc_3 - vc_0) * (y_v1 - y_v3) +
                         (va[4, :] - va[5, :]) * (yc[2] - yc_0)) / a[2]
        dvdy_e3 = -0.5 * ((vc_3 - vc_0) * (x_v1 - x_v3) +
                          (va[4, :] - va[5, :]) * (xc[2] - xc_0)) / a[2]

        dvdx_c0 = (a[0] * dvdx_e1 + a[1] * dvdx_e2 +
                   a[2] * dvdx_e3) / np.sum(a, axis=0)
        dvdy_c0 = (a[0] * dvdy_e1 + a[1] * dvdy_e2 +
                   a[2] * dvdy_e3) / np.sum(a, axis=0)

        dvdx_c = [dvdx_c0[e] for e in (e1, e2, e3)]
        dvdy_c = [dvdy_c0[e] for e in (e1, e2, e3)]

        # Use face gradients at ghost elements.
        dvdx_e = [dvdx_e1, dvdx_e2, dvdx_e3]
        dvdy_e = [dvdy_e1, dvdy_e2, dvdy_e3]
        for i in range(3):
            dvdx_c[i][id[i]] = dvdx_e[i][id[i]]
            dvdy_c[i][id[i]] = dvdy_e[i][id[i]]

        # Build weights used in limiting.
        g1, g2, g3 = [dvdx_c_i**2 + dvdy_c_i **
                      2 for dvdx_c_i, dvdy_c_i in zip(dvdx_c, dvdy_c)]
        fac = g1**2 + g2**2 + g3**2
        w = ((g2 * g3 + eps) / (fac + 3.0 * eps), (g1 * g3 + eps) /
             (fac + 3.0 * eps), (g1 * g2 + eps) / (fac + 3.0 * eps))

        # Limit gradients.
        l_dvdx_c0 = np.sum(
            [w_i * dvdx_c_i for w_i, dvdx_c_i in zip(w, dvdx_c)], axis=0)
        l_dvdy_c0 = np.sum(
            [w_i * dvdy_c_i for w_i, dvdy_c_i in zip(w, dvdy_c)], axis=0)

        # Evaluate limited gradient and cell averages at all nodes of each element.
        d_v[..., n] = dx * np.tile(l_dvdx_c0, (config.n_p, 1)) + dy * np.tile(l_dvdy_c0, (config.n_p, 1))
        a_v[..., n] = np.tile(vc_0, (config.n_p, 1))

    # 5. Reconstruct conserved variables using cell averages and limited gradients.
    ave_rho, ave_u, ave_v, ave_p = [a_v[..., i] for i in range(4)]
    d_rho, d_u, d_v, d_p = [d_v[..., i] for i in range(4)]

    # Reconstruct and check for small densities and/or pressures.
    tol = 1e-2

    l_rho = ave_rho + d_rho
    ids = np.where(np.min(l_rho, axis=0) < tol)
    while len(ids) > 0:
        print('Correcting negative density at {} locations.'.format(len(ids)))
        d_rho[:, ids] *= 0.5
        l_rho = ave_rho + d_rho
        ids = np.where(np.min(l_rho, axis=0) < tol)

    # Reconstruct momentum.
    l_rhou = ave_rhou + ave_rho * d_u + d_rho * ave_u
    l_rhov = ave_rhov + ave_rho * d_v + d_rho * ave_v

    # Reconstruct energy.
    d_ener = (1.0 / (gamma - 1.0)) * d_p + 0.5 * d_rho * (ave_u**2 + ave_v**2) + ave_rho * (ave_u * d_u + ave_v * d_v)
    l_ener = ave_ener + d_ener

    # Check for negative pressure and change it to zero gradient.
    l_p = (gamma - 1.0) * (l_ener - 0.5 * (l_rhou**2 + l_rhov**2) / l_rho)
    ids = np.where(np.min(l_p, axis=0) < tol)
    if len(ids) > 0:
        print('Correcting negative pressure at {} locations.'.format(len(ids)))
        l_p[:, ids] = ave_ener[:, ids]

    # Replace limited gradients with face gradient at boundary faces.
    l_q = np.zeros(config.n_p, config.k, 4)
    l_q[..., 0], l_q[..., 1], l_q[..., 2], l_q[..., 3] = l_rho, l_rhou, l_rhov, l_ener

    return l_q