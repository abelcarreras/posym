import numpy as np
from posym.operations import Operation


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


def get_perpendicular(vector):
    for i, v in enumerate(vector):
        if np.abs(v) < 1e-5:
            v1 = np.zeros_like(vector)
            v1[i] = 1
            v2 = np.cross(vector, v1).tolist()

            return v1, v2

    for i, v in enumerate(vector):
        if np.abs(v) >= 1e-5:

            vr = np.array(vector).copy()
            vr[i] = -vr[i]

            v1 = np.cross(vector, vr).tolist()
            v2 = np.cross(vector, v1).tolist()

            return v1, v2


def reflection(reflection_axis):

    reflection_axis = np.array(reflection_axis) / np.linalg.norm(reflection_axis)

    axis1, axis2 = get_perpendicular(reflection_axis)

    return np.outer(axis1, axis1) + np.outer(axis2, axis2) - np.outer(reflection_axis, reflection_axis)


class Reflection(Operation):
    def __init__(self, coordinates, modes, axis, symbols=None):
        super().__init__(coordinates, symbols)

        self._axis = axis

        #self._measure_mode = []
        #self._measure_coor = []

        operation = reflection(self._axis)
        operated_coor = np.dot(operation, self._coordinates.T).T

        for mode in modes:

            operated_mode = np.dot(operation, prepare_vector(self._coordinates, mode).T).T - operated_coor
            norm_1 = np.linalg.norm(mode, axis=1)

            mesure_coor, permu  = self.get_permutation(operation)

            permu_mode = np.array(operated_mode)[permu]
            norm_2 = np.linalg.norm(permu_mode, axis=1)

            self._measure_mode.append(np.average(np.divide(np.diag(np.dot(mode, permu_mode.T)), norm_1 * norm_2)))
            self._measure_coor.append(mesure_coor)

