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
                     int(self._order),
                     int(self._exp)
                     ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_measure_modes(self, coordinates, modes, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_mode = []
        # for angle in np.linspace(2*np.pi/self._order, 2*np.pi, self._order)[:-1]:
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:

            operation = rotation(angle, rotated_axis)
            mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

            measure_mode_list = []
            for mode in modes:
                operated_mode = np.dot(operation, np.array(mode).T).T
                permu_mode = np.array(operated_mode)[permu]
                measure_mode_list.append(np.trace(np.dot(mode, permu_mode.T))/np.linalg.norm(mode))

            measure_mode.append(measure_mode_list)

        measure_mode_total = np.average(measure_mode, axis=0)

        return measure_mode_total

    def get_measure_atom(self, coordinates, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)
        measure_atoms = np.array([1 if i == p else 0 for i, p in enumerate(permu)])

        return np.sum(measure_atoms)

    def get_measure_xyz(self, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_mode = []
        # for angle in np.linspace(2*np.pi/self._order, 2*np.pi, self._order)[:-1]:
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:

            operation = rotation(angle, rotated_axis)

            measure_mode_list = []
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                operated_axis = np.dot(operation, axis)
                measure_mode_list.append(np.dot(axis, operated_axis))

            measure_mode.append(measure_mode_list)

        measure_mode_total = np.average(measure_mode, axis=0)

        return np.sum(measure_mode_total)

    def get_measure_pos(self, coordinates, symbols, orientation=None, normalized=True):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_coor = []
        # for angle in np.linspace(2*np.pi/self._order, 2*np.pi, self._order)[:-1]:
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
            operation = rotation(angle, rotated_axis)

            mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)
            measure_coor.append(mesure_coor)

        measure_coor_total = np.average(measure_coor)

        if normalized:
            measure_coor_total /= np.einsum('ij, ij -> ', coordinates, coordinates)

        return measure_coor_total

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        fn_function_r = op_function1.copy()
        fn_function_r.apply_linear_transformation(operation)

        return (op_function2*fn_function_r).integrate

    def apply_rotation(self, orientation):
        self._axis = orientation.apply(self._axis)

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def exp(self):
        return self._exp

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
                op_list.append(Rotation(self._label, new_axis, self._order, self._exp))

            return op_list
