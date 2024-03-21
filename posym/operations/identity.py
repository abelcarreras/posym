from posym.operations import Operation
import numpy as np


class Identity(Operation):
    def __init__(self, label):
        super().__init__(label)

    def __str__(self):
        return 'SymOp.Identity {} <{}>'.format(self._label, hex(id(self)))

    def get_measure_modes(self, modes, orientation=None):
        self._measure_mode = [1.0] * len(modes)

        return np.array(self._measure_mode)

    def get_measure_atom(self, orientation=None):
        return len(self.permutation)

    def get_measure_xyz(self, orientation=None):
        return 3

    def get_displacements_projection(self, orientation=None):
        n_atoms = len(self.permutation)
        return np.identity(3*n_atoms).reshape(3*n_atoms, n_atoms, 3)#.tolist()

    def get_operated_coordinates(self, coordinates, orientation=None):
        return np.array(coordinates)

    def get_overlap_func(self, op_function1, op_function2, orientation=None):
        return (op_function1*op_function2).integrate

    def get_measure_pos(self, coordinates, orientation=None, normalized=True):
        if normalized:
            return 1.0
        else:
            return np.einsum('ij, ij -> ', coordinates, coordinates)

    def apply_rotation(self, orientation):
        pass

    @property
    def matrix_representation(self):
        return np.identity(3)

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            from posym.operations.products import get_operation_from_matrix
            matrix_product = self.matrix_representation @ other.matrix_representation
            return get_operation_from_matrix(matrix_product)

