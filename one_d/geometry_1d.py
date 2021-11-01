import itertools
import numpy as np
import scipy.sparse as sp_sparse


def connect_1d(e_to_v):
    """Builds global connectivity arrays for 1D grid."""
    n_faces = 2

    # Find number of elements and vertices.
    k = e_to_v.shape[0]
    n_faces_tot = n_faces * k
    n_v = k + 1

    # List of local face to local vertex connections.
    v_n = [0, 1]

    # Build global face to node sparse array.
    indices = np.array([[sk, e_to_v[idx[0], idx[1]]] for sk, idx in \
            enumerate(itertools.product(range(k), range(n_faces)))])
    sp_f_to_v = sp_sparse.csr_matrix((np.ones((n_faces_tot,)), \
            tuple(indices.T)))

    # Build global fce to global face sparse array.
    sp_f_to_f = sp_f_to_v.dot(sp_f_to_v.T) - sp_sparse.identity(n_faces_tot)

    # Find complet face to face connections.
    faces = np.argwhere(sp_f_to_f == 1)

    # Convert face global number to element and face numbers.
    element = [(int(faces_i[0] // n_faces), int(faces_i[1] // n_faces)) \
            for faces_i in faces]
    face = [(int(faces_i[0] % n_faces), int(faces_i[1] % n_faces)) \
            for faces_i in faces]

    # Rearrange into n_elements x n_faces sized arrays.
    e_to_e = np.matmul(np.reshape(np.arange(k), (k, 1)), \
            np.ones((1, n_faces)))
    e_to_f = np.matmul(np.ones((k, 1)), \
            np.reshape(np.arange(n_faces), (1, n_faces)))
    for i in range(len(faces)):
        e_to_e[element[i][0], face[i][0]] = int(element[i][1])
        e_to_f[element[i][0], face[i][0]] = int(face[i][1])

    return e_to_e, e_to_f

