"""A library that builds the connectivity arrays for 2D simulations."""

import itertools
import numpy as np
import scipy.sparse as sp_sparse


def connect(e_to_v):
    """Builds global connectivity arrays based on standard e_to_v."""
    # Only triangular element is handled in this function.
    n_faces = 3

    # Find the number of elements, vertices, and faces.
    k = e_to_v.shape[0]
    n_v = np.max(e_to_v)
    n_faces_tot = n_faces * k

    # Create the list of local face to local vertex connections.
    v_n = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)

    # Build global face to node sparse array.
    indices = np.array( \
            [[[sk, e_to_v[idx[0], v_n[idx[1], i]]] for i in range(2)] \
            for sk, idx in \
            enumerate(itertools.product(range(k), range(n_faces)))])
    indices = np.reshape(indices, (2 * n_faces_tot, 2))
    sp_f_to_v = sp_sparse.csr_matrix( \
            (np.ones((2 * n_faces_tot,)), tuple(indices.T)))

    # Build global face to global face sparse array.
    sp_f_to_f = sp_f_to_v.dot(sp_f_to_v.T) - \
            2.0 * sp_sparse.identity(n_faces_tot)

    # Find complete face to face connections.
    faces = np.argwhere(sp_f_to_f == 2)

    # Convert face global number to element and face numbers.
    element = [(int(faces_i[0] // n_faces), int(faces_i[1] // n_faces)) \
            for faces_i in faces]
    face = [(int(faces_i[0] % n_faces), int(faces_i[1] % n_faces)) \
            for faces_i in faces]

    # Rearrange into n_elements x n_faces_tots sized arrays.
    e_to_e = np.matmul(np.reshape(np.arange(k), (k, 1)), \
            np.ones((1, n_faces)))
    e_to_f = np.matmul(np.ones((k, 1)), \
            np.reshape(np.arange(n_faces), (1, n_faces)))
    for i in range(len(faces)):
        e_to_e[element[i][0], face[i][0]] = int(element[i][1])
        e_to_f[element[i][0], face[i][0]] = int(face[i][1])

    return e_to_e.astype(int), e_to_f.astype(int)
