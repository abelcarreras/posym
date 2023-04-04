from posym.operations import Operation
import numpy as np


class Identity(Operation):
    def __init__(self, label):
        super().__init__(label)

    def __hash__(self):
        return hash((self._label))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_measure_modes(self, coordinates, modes, symbols, orientation=None):
        self._measure_mode = [1.0] * len(modes)

        return np.array(self._measure_mode)

    def get_measure_atom(self, coordinates, symbols, orientation=None):
        return len(coordinates)

    def get_measure_xyz(self, orientation=None):
        return 3

    def get_overlap_func(self, op_function1, op_function2, orientation=None):
        return (op_function1*op_function2).integrate

    def get_measure_pos(self, coordinates, symbols, orientation=None, normalized=True):
        if normalized:
            return 1.0
        else:
            return np.einsum('ij, ij -> ', coordinates, coordinates)

    def apply_rotation(self, orientation):
        pass

    @property
    def operation_matrix_list(self):
        return [np.identity(3)]

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            return [Identity(self._label)]

