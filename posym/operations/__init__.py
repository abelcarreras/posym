import numpy as np
from functools import lru_cache


@lru_cache(maxsize=100)
def get_submatrix_indices(symbols):
    # separate distance_table in submatrices corresponding to a single symbol
    submatrices_indices = []
    for s in np.unique(symbols):
        submatrices_indices.append([j for j, s2 in enumerate(symbols) if s2 == s])

    return submatrices_indices


def get_permutation_hungarian(distance_table, symbols):
    """
    This function takes distance_table and returns the permutation vector
    that minimizes its trace, using the Hungarian method.
    """
    from scipy.optimize import linear_sum_assignment

    def get_perm_submatrix(sub_matrix):
        row_ind, col_ind = linear_sum_assignment(sub_matrix)
        perm = np.zeros_like(row_ind)
        perm[row_ind] = col_ind
        return perm

    submatrices_indices = get_submatrix_indices(symbols)

    # determine the permutation for each submatrix
    perm_submatrices = []
    for index in submatrices_indices:
        submatrix = np.array(distance_table)[index, :][:, index]
        perm_sub = get_perm_submatrix(submatrix)
        perm_submatrices.append(perm_sub)

    # restore global permutation by joining permutations of submatrices
    global_permutation = np.zeros(len(distance_table), dtype=int)
    for index, perm in zip(submatrices_indices, perm_submatrices):
        index = np.array(index)
        global_permutation[index] = index[perm]

    return list(global_permutation)


class Operation:
    def __init__(self, label):
        self._label = label

    def get_permutation(self, operation, coordinates, symbols, return_dot=False):

        operated_coor = np.dot(operation, coordinates.T).T
        symbols = tuple(int.from_bytes(num.encode(), 'big') for num in symbols)

        dot_table = -np.dot(coordinates, operated_coor.T)
        permutation = get_permutation_hungarian(dot_table, symbols)

        if return_dot:
            permu_coor = operated_coor[permutation]
            measure = np.einsum('ij, ij -> ', coordinates, permu_coor)
            # measure = np.trace(np.dot(coordinates, permu_coor.T))
            return measure, permutation
        else:
            return permutation

    def get_normalization(self, coordinates):

        sum_list = []
        for r1 in coordinates:
            for r2 in coordinates:
                subs = np.subtract(r1, r2)
                sum_list.append(np.dot(subs, subs))
        d = np.average(sum_list)

        return d

    @property
    def label(self):
        return self._label
