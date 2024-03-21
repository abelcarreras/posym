from posym.operations import Operation
import numpy as np


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


def inversion():
    return -np.identity(3)


class Inversion(Operation):
    def __init__(self, label):
        super().__init__(label)
        self._order = 1
        self._determinant = -1

    def __str__(self):
        return 'SymOp.Inversion {} <{}>'.format(self._label, hex(id(self)))

    def get_measure_modes(self, modes, orientation=None):

        operation = inversion()

        measure_mode = []
        for mode in modes:
            operated_mode = np.dot(operation, np.array(mode).T).T
            norm = np.linalg.norm(mode)
            permu_mode = np.array(operated_mode)[self.permutation]

            measure_mode.append(np.trace(np.dot(mode, permu_mode.T))/norm)

        return np.array(measure_mode)

    def get_measure_atom(self):

        measure_atoms = np.array([1 if i == p else 0 for i, p in enumerate(self.permutation)])

        return np.sum(measure_atoms)

    def get_measure_xyz(self, orientation=None):

        operation = inversion()

        measure_mode = []
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            operated_axis = np.dot(operation, axis)
            measure_mode.append(np.dot(axis, operated_axis))

        return np.sum(measure_mode)

    def get_displacements_projection(self, orientation=None):

        n_atoms = len(self.permutation)
        operation = inversion()

        projected_modes = []
        cartesian_modes = np.identity(3 * n_atoms).reshape(3 * n_atoms, n_atoms, 3)

        for i, mode in enumerate(cartesian_modes):
            operated_mode = np.dot(operation, np.array(mode).T).T
            projected_modes.append(operated_mode[self.permutation])

        return np.array(projected_modes)

    def get_measure_pos(self, coordinates, orientation=None, normalized=True):

        operation = inversion()
        operated_coor = np.dot(operation, coordinates.T).T
        mesure_pos = np.einsum('ij, ij -> ', coordinates, operated_coor[self.permutation])

        if normalized:
            mesure_pos /= np.einsum('ij, ij -> ', coordinates, coordinates)

        return mesure_pos

    def get_operated_coordinates(self, coordinates, orientation=None):

        operation = inversion()
        operated_coor = np.dot(operation, coordinates.T).T

        return operated_coor[self.permutation]

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        operation = inversion()

        op_function_i = op_function1.copy()
        op_function_i.apply_linear_transformation(operation)

        return (op_function2*op_function_i).integrate

    def apply_rotation(self, orientation):
        pass

    @property
    def matrix_representation(self):
        return inversion()
