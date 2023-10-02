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


def cache_permutation(func):
    cache_dict = {}

    def wrapper_cache(self, operation, coordinates, symbols):

        hash_key = (np.array2string(operation), np.array2string(coordinates), tuple(symbols))
        if hash_key in cache_dict:
            return cache_dict[hash_key]

        cache_dict[hash_key] = func(self, operation, coordinates, symbols)
        return cache_dict[hash_key]

    return wrapper_cache


class Operation:
    def __init__(self, label):
        self._label = label

    @cache_permutation
    def _get_permutation(self, operation, coordinates, symbols):
        operated_coor = np.dot(operation, coordinates.T).T
        symbols = tuple(int.from_bytes(num.encode(), 'big') for num in symbols)

        dot_table = -np.dot(coordinates, operated_coor.T)
        permutation = get_permutation_hungarian(dot_table, symbols)

        return permutation

    def _get_operated_coordinates(self, operation, coordinates, symbols):
        """
        get coordinates operated and permuted

        :param operation: operator matrix representation (3x3)
        :param coordinates: coordinates to be operated
        :param symbols: atomic symbols
        :return: operated and permuted coordinates
        """
        operated_coor = np.dot(operation, coordinates.T).T
        permutation = self._get_permutation(operation, coordinates, symbols)
        permu_coor = operated_coor[permutation]
        return permu_coor

    @property
    def label(self):
        return self._label
