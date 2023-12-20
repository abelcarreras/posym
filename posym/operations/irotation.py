from posym.operations import Operation
from posym.operations.rotation import rotation
from posym.operations.reflection import reflection
from posym.tools import standardize_vector
from posym.operations.permutation import roll_permutation
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
                     int(self._order),
                     int(self._exp)
                     ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_measure_modes(self, coordinates, modes, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_mode = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
        # for angle in np.arange(2*np.pi/self._order, 2*np.pi, 2*np.pi/self._order)[::2]:
            operation1 = rotation(angle, rotated_axis)
            operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
            operation = np.dot(operation2, operation1)

            operated_coor = np.dot(operation, coordinates.T).T

            permu = self._get_permutation(operation, coordinates, symbols)

            measure_mode_list = []
            for mode in modes:

                operated_mode = np.dot(operation, prepare_vector(coordinates, mode).T).T - operated_coor
                norm = np.linalg.norm(mode)
                permu_mode = np.array(operated_mode)[permu]

                measure_mode_list.append(np.trace(np.dot(mode, permu_mode.T))/norm)

            measure_mode.append(measure_mode_list)

        measure_mode_total = np.average(measure_mode, axis=0)

        return measure_mode_total

    def get_measure_atom(self, coordinates, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        permu = self._get_permutation(operation, coordinates, symbols)
        measure_atoms = np.array([1 if i == p else 0 for i, p in enumerate(permu)])

        return np.sum(measure_atoms)

    def get_measure_xyz(self, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        measure_mode = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:

            operation1 = rotation(angle, rotated_axis)
            operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
            operation = np.dot(operation2, operation1)

            measure_mode_list = []
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                operated_axis = np.dot(operation, axis)
                measure_mode_list.append(np.dot(axis, operated_axis))

            measure_mode.append(measure_mode_list)

        measure_mode_total = np.average(measure_mode, axis=0)

        return np.sum(measure_mode_total)

    def get_displacements_projection(self, coordinates, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        total_project = np.zeros_like(coordinates)
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:

            operation1 = rotation(angle, rotated_axis)
            operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
            operation = np.dot(operation2, operation1)

            permu = self._get_permutation(operation, coordinates, symbols)

            cartesian_modes = np.identity(3 * len(symbols)).reshape(3 * len(symbols), len(symbols), 3)

            projected_modes = []
            for i, mode in enumerate(cartesian_modes):
                operated_mode = np.dot(operation, np.array(mode).T).T
                projected_modes.append(operated_mode[permu])

            total_project += np.array(projected_modes)

        return total_project/2

    def get_measure_pos(self, coordinates, symbols, orientation=None, normalized=True):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order # * self._exp
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        permutation = self._get_permutation(operation, coordinates, symbols)
        permutation = roll_permutation(permutation, self._exp)

        angle = 2 * np.pi / self._order * self._exp
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        operated_coor = np.dot(operation, coordinates.T).T

        measure_coor_total = np.einsum('ij, ij -> ', coordinates, operated_coor[permutation])

        if normalized:
            measure_coor_total /= np.einsum('ij, ij -> ', coordinates, coordinates)

        return measure_coor_total

    def get_operated_coordinates(self, coordinates, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis)
        operation = np.dot(operation2, operation1)

        permutation = self._get_permutation(operation, coordinates, symbols)

        permutation = roll_permutation(permutation, self._exp)

        operated_coor_perm = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
            operation1 = rotation(angle, rotated_axis)
            operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
            operation = np.dot(operation2, operation1)

            operated_coor = np.dot(operation, coordinates.T).T
            operated_coor_perm.append(operated_coor[permutation])

            if self._order / self._exp == 2:
                break

            permutation = np.argsort(permutation)

        return operated_coor_perm

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        op_function_ir = op_function1.copy()
        op_function_ir.apply_linear_transformation(operation)

        return (op_function2*op_function_ir).integrate

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

        op_matrix_list = []
        for angle in [2 * np.pi / self._order * self._exp, -2 * np.pi / self._order * self._exp]:
            operation1 = rotation(angle, self._axis)
            operation2 = reflection(self._axis) if np.mod(self._exp, 2) != 0 else np.identity(3)
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
                op_list.append(ImproperRotation(self._label, new_axis, self._order, self._exp))

            return op_list
