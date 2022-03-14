from posym.operations import Operation
from posym.operations.rotation import rotation
from posym.operations.reflection import reflection
from posym.tools import standardize_vector
import numpy as np


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


class ImproperRotation(Operation):
    def __init__(self, label, axis, order=1, exp=1):
        super().__init__(label)

        self._axis = standardize_vector(axis)
        self._order = order
        self._exp = exp

    def __hash__(self):
        axis = np.round(self._axis, decimals=6)
        axis[axis == 0.] = 0.

        return hash((self._label,
                     np.array2string(axis),
                     int(self._order)
                     ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_measure(self, coordinates, modes, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_mode = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
        # for angle in np.arange(2*np.pi/self._order, 2*np.pi, 2*np.pi/self._order)[::2]:
            operation1 = rotation(angle, rotated_axis)
            operation2 = reflection(rotated_axis)
            operation = np.dot(operation2, operation1)

            operated_coor = np.dot(operation, coordinates.T).T

            mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

            measure_mode_list = []
            for mode in modes:

                operated_mode = np.dot(operation, prepare_vector(coordinates, mode).T).T - operated_coor
                norm = np.linalg.norm(mode)
                permu_mode = np.array(operated_mode)[permu]

                measure_mode_list.append(np.trace(np.dot(mode, permu_mode.T))/norm)

            measure_mode.append(measure_mode_list)

        measure_mode_total = np.average(measure_mode, axis=0)

        return measure_mode_total

    def get_measure_pos(self, coordinates, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_coor = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
        # for angle in np.arange(2*np.pi/self._order, 2*np.pi, 2*np.pi/self._order)[::2]:
            operation1 = rotation(angle, rotated_axis)
            operation2 = reflection(rotated_axis)
            operation = np.dot(operation2, operation1)

            mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

            measure_coor.append(mesure_coor)

        measure_coor_total = np.average(measure_coor)

        return measure_coor_total

    def get_measure_func(self, op_function, self_similarity, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_fn = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
            operation1 = rotation(angle, rotated_axis)
            operation2 = reflection(rotated_axis)
            operation = np.dot(operation2, operation1)

            op_function_ir = op_function.copy()
            op_function_ir.apply_linear_transformation(operation)

            measure_fn.append((op_function_ir*op_function).integrate/self_similarity)

        measure_coor_total = np.average(measure_fn)

        return measure_coor_total


    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def operation_matrix_list(self):

        op_matrix_list = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
            operation1 = rotation(angle, self._axis)
            operation2 = reflection(self._axis)
            operation = np.dot(operation2, operation1)
            op_matrix_list.append(operation)
        return op_matrix_list

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            op_list = []
            for op_mat in other.operation_matrix_list:
                new_axis = np.dot(op_mat, self._axis)
                op_list.append(ImproperRotation(self._label, new_axis, self._order))

            return op_list
