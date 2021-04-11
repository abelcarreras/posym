from posym.operations import Operation
from scipy.spatial.transform import Rotation as R
import numpy as np


def rotation(angle, rotation_axis):

    rotation_vector = angle * np.array(rotation_axis) / np.linalg.norm(rotation_axis)
    rotation = R.from_rotvec(rotation_vector)

    return rotation.as_matrix()


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


class Rotation(Operation):
    def __init__(self, coordinates, modes, axis, order, symbols=None):
        super().__init__(coordinates, symbols)

        self._axis = axis
        self._order = order

        for angle in np.arange(2*np.pi/order, 2*np.pi, 2*np.pi/order):
            operation = rotation(angle, self._axis)
            operated_coor = np.dot(operation, self._coordinates.T).T

            measure_mode_list = []
            measure_coor_list = []

            for mode in modes:

                operated_mode = np.dot(operation, prepare_vector(self._coordinates, mode).T).T - operated_coor
                norm_1 = np.linalg.norm(mode, axis=1)

                mesure_coor, permu  = self.get_permutation(operation)

                permu_mode = np.array(operated_mode)[permu]
                norm_2 = np.linalg.norm(permu_mode, axis=1)

                measure_mode_list.append(np.average(np.divide(np.diag(np.dot(mode, permu_mode.T)), norm_1 * norm_2)))
                measure_coor_list.append(mesure_coor)

            self._measure_mode.append(measure_mode_list)
            self._measure_coor.append(measure_coor_list)

        self._measure_mode = np.average(self._measure_mode, axis=0)
        self._measure_coor = np.average(self._measure_coor, axis=0)
