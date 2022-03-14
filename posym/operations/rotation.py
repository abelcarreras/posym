from posym.operations import Operation
from scipy.spatial.transform import Rotation as R
from posym.tools import standardize_vector
import numpy as np
import hashlib


def rotation(angle, rotation_axis):

    rotation_vector = angle * np.array(rotation_axis) / np.linalg.norm(rotation_axis)
    rotation = R.from_rotvec(rotation_vector)

    return rotation.as_matrix()


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


class Rotation(Operation):
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

        #print(self._measure_mode)

        measure_mode = []

        # print('llñ', list(np.linspace(2*np.pi/self._order, 2*np.pi, self._order)))
        # for angle in np.linspace(2*np.pi/self._order, 2*np.pi, self._order)[:-1]:
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:

            operation = rotation(angle, rotated_axis)
            # operated_coor = np.dot(operation, coordinates.T).T

            mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

            measure_mode_list = []
            for mode in modes:

                # operated_mode = np.dot(operation, prepare_vector(coordinates, mode).T).T - operated_coor
                operated_mode = np.dot(operation, np.array(mode).T).T

                norm = np.linalg.norm(mode)

                permu_mode = np.array(operated_mode)[permu]

                measure_mode_list.append(np.trace(np.dot(mode, permu_mode.T))/norm)

            measure_mode.append(measure_mode_list)

        measure_mode_total = np.average(measure_mode, axis=0)

        return measure_mode_total

    def get_measure_pos(self, coordinates, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_coor = []
        # for angle in np.linspace(2*np.pi/self._order, 2*np.pi, self._order)[:-1]:
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
            operation = rotation(angle, rotated_axis)

            mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)
            measure_coor.append(mesure_coor)

        measure_coor_total = np.average(measure_coor)

        return measure_coor_total

    def get_measure_func(self, op_function, self_similarity, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_op = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
            operation = rotation(angle, rotated_axis)

            fn_function_r = op_function.copy()
            fn_function_r.apply_linear_transformation(operation)

            measure_op.append((fn_function_r*op_function).integrate/self_similarity)

        measure_coor_total = np.average(measure_op)

        return measure_coor_total


    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def operation_matrix_list(self):
        return [rotation(angle, self._axis) for angle in
                [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]]

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            op_list = []
            for op_mat in other.operation_matrix_list:
                new_axis = np.dot(op_mat, self._axis)
                op_list.append(Rotation(self._label, new_axis, self._order))

            return op_list
