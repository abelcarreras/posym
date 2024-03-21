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
        self._exp = np.mod(exp, order)
        self._determinant = -1

    def __str__(self):
        axis_txt = '[{:8.3f} {:8.3f} {:8.3f}]'.format(*self._axis)
        return 'SymOp.ImproperRotation {} {} order: {} exp: {} <{}>'.format(self._label, axis_txt, self._order, self._exp, hex(id(self)))

    def get_measure_modes(self, modes, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order * self._exp

        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis) # if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        measure_mode = []
        for mode in modes:
            operated_mode = np.dot(operation, np.array(mode).T).T
            norm = np.linalg.norm(mode)
            permu_mode = np.array(operated_mode)[self.permutation]
            measure_mode.append(np.trace(np.dot(mode, permu_mode.T))/norm)

        return measure_mode

    def get_measure_atom(self):

        measure_atoms = np.array([1 if i == p else 0 for i, p in enumerate(self.permutation)])

        return np.sum(measure_atoms)

    def get_measure_xyz(self, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp

        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis)  #if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        measure_mode = []
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            operated_axis = np.dot(operation, axis)
            measure_mode.append(np.dot(axis, operated_axis))

        return np.sum(measure_mode)

    def get_displacements_projection(self, orientation=None):

        n_atoms = len(self.permutation)
        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp

        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis) # if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        cartesian_modes = np.identity(3 * n_atoms).reshape(3 * n_atoms, n_atoms, 3)

        projected_modes = []
        for i, mode in enumerate(cartesian_modes):
            operated_mode = np.dot(operation, np.array(mode).T).T
            projected_modes.append(operated_mode[self.permutation])

        return projected_modes

    def get_measure_pos(self, coordinates, orientation=None, normalized=True):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order * self._exp
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis)
        operation = np.dot(operation2, operation1)

        operated_coor = np.dot(operation, coordinates.T).T
        mesure_pos = np.einsum('ij, ij -> ', coordinates, operated_coor[self.permutation])

        if normalized:
            mesure_pos /= np.einsum('ij, ij -> ', coordinates, coordinates)

        return mesure_pos

    def get_operated_coordinates(self, coordinates, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order * self._exp
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis)
        operation = np.dot(operation2, operation1)

        operated_coor = np.dot(operation, coordinates.T).T
        return operated_coor[self.permutation]

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp
        operation1 = rotation(angle, rotated_axis)
        operation2 = reflection(rotated_axis) #if np.mod(self._exp, 2) != 0 else np.identity(3)
        operation = np.dot(operation2, operation1)

        op_function_ir = op_function1.copy()
        op_function_ir.apply_linear_transformation(operation)

        return (op_function2*op_function_ir).integrate

    def apply_rotation(self, orientation):
        self._axis = orientation.apply(self._axis)

    def inverse(self):
        return ImproperRotation(self.label, axis=self._axis, order=self._order, exp=-self._exp)

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
    def matrix_representation(self):

        angle = 2 * np.pi / self._order * self._exp
        operation1 = rotation(angle, self._axis)
        operation2 = reflection(self._axis)
        operation = np.dot(operation2, operation1)

        return operation
