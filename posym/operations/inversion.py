from posym.operations import Operation
import numpy as np


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


def inversion():
    return -np.identity(3)


class Inversion(Operation):
    def __init__(self, coordinates, modes):
        super().__init__(coordinates)

        self._measure_mode = []
        self._measure_coor = []

        operation = inversion()
        operated_coor = np.dot(operation, self._coordinates.T).T

        for mode in modes:

            operated_mode = np.dot(operation, prepare_vector(self._coordinates, mode).T).T - operated_coor
            norm_1 = np.linalg.norm(mode, axis=1)

            mesure_coor, permu  = self.get_permutation(operation)

            permu_mode = np.array(operated_mode)[permu]
            norm_2 = np.linalg.norm(permu_mode, axis=1)

            self._measure_mode.append(np.average(np.divide(np.diag(np.dot(mode, permu_mode.T)), norm_1 * norm_2)))
            self._measure_coor.append(mesure_coor)

