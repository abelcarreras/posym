from functools import lru_cache
# from posym.permutations import get_cross_distance_table
from scipy.optimize import linear_sum_assignment
import numpy as np


@lru_cache(maxsize=100)
def get_submatrix_indices(symbols):
    # separate distance_table in submatrices corresponding to a single symbol
    submatrices_indices = []
    for s in np.unique(symbols):
        submatrices_indices.append([j for j, s2 in enumerate(symbols) if s2 == s])

    return submatrices_indices


def get_permutation_labels(distance_table, symbols, permutation_function):
    """
    This function restricts permutations by the use of atom labels
    returns the permutation vector that minimizes its trace using custom algorithms.
    """
    submatrices_indices = get_submatrix_indices(symbols)

    # determine the permutation for each submatrix
    perm_submatrices = []
    for index in submatrices_indices:
        submatrix = np.array(distance_table)[index, :][:, index]
        perm_sub = permutation_function(submatrix)
        perm_submatrices.append(perm_sub)

    # restore global permutation by joining permutations of submatrices
    global_permutation = np.zeros(len(distance_table), dtype=int)
    for index, perm in zip(submatrices_indices, perm_submatrices):
        index = np.array(index)
        global_permutation[index] = index[perm]

    return np.array(global_permutation)


def cache_permutation(func):
    cache_dict = {}

    def wrapper_cache(operation, coordinates, symbols):
        hash_key = (np.array2string(operation), np.array2string(coordinates), tuple(symbols))
        if hash_key in cache_dict:
            return cache_dict[hash_key]

        cache_dict[hash_key] = func(operation, coordinates, symbols)
        return cache_dict[hash_key]

    return wrapper_cache


@cache_permutation
def get_permutation_hungarian(operation, coordinates, symbols):

    operated_coor = np.dot(operation, coordinates.T).T
    symbols = tuple(int.from_bytes(num.encode(), 'big') for num in symbols)

    dot_table = -np.dot(coordinates, operated_coor.T)
    # dot_table = get_cross_distance_table(coordinates, operated_coor)

    # permutation algorithms functions
    def hungarian_algorithm(sub_matrix):
        row_ind, col_ind = linear_sum_assignment(sub_matrix)
        perm = np.zeros_like(row_ind)
        perm[row_ind] = col_ind
        return perm

    return get_permutation_labels(dot_table, symbols, hungarian_algorithm)
