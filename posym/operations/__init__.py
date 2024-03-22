import numpy as np
from functools import lru_cache
# from posym.permutations import get_cross_distance_table
from posym.permutations import get_permutation_annealing, get_permutation_brute, fix_permutation, validate_permutation  # noqa
from scipy.optimize import linear_sum_assignment
from posym.config import Configuration, CustomPerm
from posym.operations.permutation import Permutation


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

    def wrapper_cache(operation, coordinates, symbols, order):
        hash_key = (np.array2string(operation), np.array2string(coordinates), tuple(symbols))
        if hash_key in cache_dict:
            return cache_dict[hash_key]

        cache_dict[hash_key] = func(operation, coordinates, symbols, order)
        return cache_dict[hash_key]

    return wrapper_cache


@cache_permutation
def get_permutation_aprox(operation, coordinates, symbols, order):

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

    def annealing_algorithm(dot_matrix):
        return get_permutation_annealing(dot_matrix, order, 1)

    def brute_force_algorithm(dot_matrix):
        return get_permutation_brute(dot_matrix, order, 1)

    # algorithms list
    algorithm_dict = {'hungarian': hungarian_algorithm,
                      'annealing': annealing_algorithm,
                      'brute_force': brute_force_algorithm}

    return get_permutation_labels(dot_table, symbols, algorithm_dict[Configuration().algorithm])


class Operation:
    def __init__(self, label):
        self._label = label
        self._order = 1
        self._exp = 1
        self._determinant = 1
        self._gen_rep = []
        self._permutation = None
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            vector = np.round(np.array(self.matrix_representation).flatten(), decimals=6)
            self._hash = hash(np.array(vector * 1e5, dtype=int).tobytes())

        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_type(self):
        normalized_exp = np.mod(self._exp, self._order)
        if normalized_exp > self._order//2:
            normalized_exp = self._order - normalized_exp

        return '{} : {} / {}'.format(type(self).__name__, abs(normalized_exp), self._order)

    def set_permutation(self, permutation):
        self._permutation = permutation

    def set_permutation_set(self, permutation_set, symbols, ignore_compatibility=False):

        def apply_op(permut, base):
            new = []
            for p, b in zip(permut, base):
                new.append(permut[b])
            return new

        n_atoms = len(symbols)
        permutation = list(range(n_atoms))
        for op in self._gen_rep:
            permutation = apply_op(permutation, permutation_set[op])

        if ignore_compatibility:
            self._permutation = permutation
            return True

        if validate_permutation(permutation, self._order, self._determinant):
            self._permutation = permutation
            return True

    def inverse(self):
        return self

    @property
    def permutation(self):
        if self._permutation is None:
            raise Exception('No permutation has been defined in', self)
        return self._permutation

    @property
    def label(self):
        return self._label

    @property
    def matrix_representation(self):
        raise NotImplementedError('Not implemented')

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            from posym.operations.products import get_operation_from_matrix, get_operation_from_matrix_test
            matrix_product = self.matrix_representation @ other.matrix_representation

            new_operator = get_operation_from_matrix(matrix_product)
            if not np.allclose(new_operator.matrix_representation, matrix_product):
                print(self, ' * ',  other, ' = ', new_operator)
                print(new_operator.matrix_representation)
                print(matrix_product)
                get_operation_from_matrix_test(matrix_product)

                raise Exception('Product error!')

            new_operator._gen_rep = list(other._gen_rep) + list(self._gen_rep)
            return new_operator
