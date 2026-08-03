from posym.operations import Operation
from scipy.spatial.transform import Rotation as Rot
from posym.tools import standardize_vector
import numpy as np


def cache_rotation(func):
    cache_dict = {}

    def wrapper_cache(angle, rotation_axis):
        hash_key = (angle, tuple(rotation_axis))
        if hash_key in cache_dict:
            return cache_dict[hash_key]

        cache_dict[hash_key] = func(angle, rotation_axis)
        return cache_dict[hash_key]

    return wrapper_cache

@cache_rotation
def rotation(angle, rotation_axis):

    rotation_vector = angle * np.array(rotation_axis) / np.linalg.norm(rotation_axis)
    rotation = Rot.from_rotvec(rotation_vector)

    return rotation.as_matrix()


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


class Rotation(Operation):
    """
    Proper rotation operation (C_n).

    Generates a rotation by :math:`2\\pi k/n` about a given axis.
    The matrix representation is a 3x3 rotation matrix with
    determinant +1.
    """
    def __init__(self, label, axis, order=1, exp=1):
        super().__init__(label)

        self._axis = standardize_vector(axis)
        self._order = order
        self._exp = np.mod(exp, order)
        self._determinant = 1

        # normalize C2
        if order <= 2:
            self._exp = abs(exp)

        if self._order > 10:
            print('ko', self._order)
            exit()

    def __str__(self):
        axis_txt = '[{:8.3f} {:8.3f} {:8.3f}]'.format(*self._axis)
        return 'SymOp.Rotation {} {} order: {} exp: {} <{}>'.format(self._label, axis_txt, self._order, self._exp, hex(id(self)))

    def get_measure_modes(self, modes, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        measure_mode = []
        for mode in modes:
            operated_mode = np.dot(operation, np.array(mode).T).T
            norm = np.linalg.norm(mode)**2
            permu_mode = np.array(operated_mode)[self.permutation]
            measure_mode.append(np.trace(np.dot(mode, permu_mode.T))/norm)

        return measure_mode

    def get_measure_modes_proj(self, modes, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        measure_mode = []
        for mode in modes:
            operated_mode = np.dot(operation, np.array(mode).T).T
            permu_mode = np.array(operated_mode)[self.permutation]

            P = np.outer(mode, mode)
            PG = np.outer(permu_mode, permu_mode)
            sigma = np.trace(P @ PG)
            measure_mode.append(sigma)

        return np.abs(measure_mode)

    def get_measure_full_proj(self, modes, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        d = len(modes)

        # Flatten modes into (3N,d)
        V = np.stack([m.reshape(-1) for m in modes], axis=1)

        # Apply symmetry operation
        operated = np.array([operation @ m.T for m in modes])  # (d,3,natoms)
        operated = np.transpose(operated, (0, 2, 1))  # (d,natoms,3)
        operated = operated[:, self.permutation]
        GV = np.stack([m.reshape(-1) for m in operated], axis=1)

        # Projectors
        P = V @ V.T
        PG = GV @ GV.T

        return np.trace(P @ PG) / d


    def get_measure_atom(self):

        measure_atoms = np.array([1 if i == p else 0 for i, p in enumerate(self.permutation)])

        return np.sum(measure_atoms)

    def get_measure_xyz(self, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        measure_mode = []
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            operated_axis = np.dot(operation, axis)
            measure_mode.append(np.dot(axis, operated_axis))

        return np.sum(measure_mode)

    def get_displacements_projection(self, orientation=None):

        n_atoms = len(self.permutation)
        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        cartesian_modes = np.identity(3 * n_atoms).reshape(3 * n_atoms, n_atoms, 3)
        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        projected_modes = []
        for i, mode in enumerate(cartesian_modes):
            operated_mode = np.dot(operation, np.array(mode).T).T
            projected_modes.append(operated_mode[self.permutation])

        return projected_modes

    def get_measure_pos(self, coordinates, orientation=None, normalized=True):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = rotation(2 * np.pi / self._order * self._exp, rotated_axis)

        operated_coor = np.dot(operation, coordinates.T).T
        mesure_pos = np.einsum('ij, ij -> ', coordinates, operated_coor[self.permutation])

        if normalized:
            mesure_pos /= np.einsum('ij, ij -> ', coordinates, coordinates)

        return mesure_pos

    def get_operated_coordinates(self, coordinates, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = rotation(2 * np.pi / self._order * self._exp, rotated_axis)

        operated_coor = np.dot(operation, coordinates.T).T

        return operated_coor[self.permutation]

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        angle = 2 * np.pi / self._order * self._exp
        operation = rotation(angle, rotated_axis)

        fn_function_r = op_function1.copy()
        fn_function_r.apply_linear_transformation(operation)

        return (op_function2*fn_function_r).integrate

    def apply_rotation(self, orientation):
        self._axis = orientation.apply(self._axis)

    def inverse(self):
        return Rotation(self.label, axis=self._axis, order=self._order, exp=-self._exp)

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
        return rotation(angle, self._axis)

