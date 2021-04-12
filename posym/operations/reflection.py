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
    def __init__(self, label, axis):
        super().__init__(label)

        self._axis = axis

    def get_measure(self, coordinates, modes, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        operation = reflection(rotated_axis)

        operated_coor = np.dot(operation, coordinates.T).T

        measure_mode = []
        #measure_coor = []

        mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

        for mode in modes:

            operated_mode = np.dot(operation, prepare_vector(coordinates, mode).T).T - operated_coor
            #norm_1 = np.dot(np.linalg.norm(mode, axis=1), np.linalg.norm(mode, axis=1))
            norm = np.linalg.norm(mode)

            permu_mode = np.array(operated_mode)[permu]

            # norm_2 = np.linalg.norm(permu_mode, axis=1)

            #self._measure_mode.append(np.nanmean(np.divide(np.diag(np.dot(mode, permu_mode.T)), norm_1 * norm_2)))
            measure_mode.append(np.add.reduce(np.diag(np.dot(mode, permu_mode.T)))/norm)
            #measure_coor.append(mesure_coor)

        return np.array(measure_mode), mesure_coor

    @property
    def axis(self):
        return self._axis