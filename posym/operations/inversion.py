from posym.operations import Operation
import numpy as np


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


def inversion():
    return -np.identity(3)


class Inversion(Operation):
    def __init__(self, label):
        super().__init__(label)
        self._order = 2

    def __hash__(self):
        return hash((self._label))

    def __eq__(self, other):
        return hash(self) == hash(other)



    def get_measure_modes(self, coordinates, modes, symbols, orientation=None):

        operation = inversion()
        operated_coor = np.dot(operation, coordinates.T).T

        permu = self._get_permutation(operation, coordinates, symbols)

        measure_mode = []
        for mode in modes:
            operated_mode = np.dot(operation, prepare_vector(coordinates, mode).T).T - operated_coor
            norm = np.linalg.norm(mode)
            permu_mode = np.array(operated_mode)[permu]

            measure_mode.append(np.trace(np.dot(mode, permu_mode.T))/norm)

        return np.array(measure_mode)

    def get_measure_atom(self, coordinates, symbols, orientation=None):

        operation = inversion()
        permu = self._get_permutation(operation, coordinates, symbols)
        measure_atoms = np.array([1 if i == p else 0 for i, p in enumerate(permu)])

        return np.sum(measure_atoms)

    def get_measure_xyz(self, orientation=None):

        operation = inversion()

        measure_mode = []
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            operated_axis = np.dot(operation, axis)
            measure_mode.append(np.dot(axis, operated_axis))

        return np.sum(measure_mode)

    def get_displacements_projection(self, coordinates, symbols, orientation=None):

        operation = inversion()

        projected_modes = []
        permu = self._get_permutation(operation, coordinates, symbols)
        cartesian_modes = np.identity(3 * len(symbols)).reshape(3 * len(symbols), len(symbols), 3)

        for i, mode in enumerate(cartesian_modes):
            operated_mode = np.dot(operation, np.array(mode).T).T
            projected_modes.append(operated_mode[permu])

        return np.array(projected_modes)

    def get_permutation_pos(self, coordinates, symbols, orientation=None):
        operation = inversion()
        return self._get_permutation(operation, coordinates, symbols)

    def get_measure_pos(self, coordinates, symbols, orientation=None, normalized=True):

        operation = inversion()
        permu_coor = self._get_operated_coordinates(operation, coordinates, symbols)
        mesure_coor = np.einsum('ij, ij -> ', coordinates, permu_coor)

        if normalized:
            normalization = np.einsum('ij, ij -> ', coordinates, coordinates)
            if abs(normalization) < 1e-10:
                return 1
            mesure_coor /= normalization

        return mesure_coor

    def get_operated_coordinates(self, coordinates, symbols, orientation=None):

        operation = inversion()
        return [self._get_operated_coordinates(operation, coordinates, symbols)]

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        operation = inversion()

        op_function_i = op_function1.copy()
        op_function_i.apply_linear_transformation(operation)

        return (op_function2*op_function_i).integrate

    def apply_rotation(self, orientation):
        pass

    @property
    def operation_matrix_list(self):
        return [inversion()]

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            return [Inversion(self._label)]
