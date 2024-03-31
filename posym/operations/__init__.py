import numpy as np
from posym.permutation.permutations import validate_permutation  # noqa


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
