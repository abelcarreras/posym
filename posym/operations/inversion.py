from posym.operations import Operation
import numpy as np


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


def inversion():
    return -np.identity(3)


class Inversion(Operation):
    def __init__(self, label):
        super().__init__(label)

    def __hash__(self):
        return hash((self._label))

    def __eq__(self, other):
        return hash(self) == hash(other)



    def get_measure(self, coordinates, modes, symbols, orientation=None):

        operation = inversion()
        operated_coor = np.dot(operation, coordinates.T).T

        mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

        measure_mode = []
        for mode in modes:
            operated_mode = np.dot(operation, prepare_vector(coordinates, mode).T).T - operated_coor
            norm = np.linalg.norm(mode)
            permu_mode = np.array(operated_mode)[permu]

            measure_mode.append(np.trace(np.dot(mode, permu_mode.T))/norm)

        return np.array(measure_mode)

    def get_measure_pos(self, coordinates, symbols, orientation=None):

        operation = inversion()
        mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

        return mesure_coor

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        operation = inversion()

        op_function_i = op_function1.copy()
        op_function_i.apply_linear_transformation(operation)

        return (op_function2*op_function_i).integrate

    @property
    def operation_matrix_list(self):
        return [inversion()]

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            return [Inversion(self._label)]
