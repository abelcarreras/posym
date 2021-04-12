from posym.operations import Operation
import numpy as np


def prepare_vector(positions, vector):
    return np.array(vector) + np.array(positions)


def inversion():
    return -np.identity(3)


class Inversion(Operation):
    def __init__(self, label):
        super().__init__(label)

    def get_measure(self, coordinates, modes, symbols, orientation=None):

        operation = inversion()
        operated_coor = np.dot(operation, coordinates.T).T

        measure_mode = []
        measure_coor = []

        mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

        for mode in modes:
            operated_mode = np.dot(operation, prepare_vector(coordinates, mode).T).T - operated_coor
            #norm_1 = np.linalg.norm(mode, axis=1)
            norm = np.linalg.norm(mode)


            permu_mode = np.array(operated_mode)[permu]
            # norm_2 = np.linalg.norm(permu_mode, axis=1)

            #self._measure_mode.append(np.nanmean(np.divide(np.diag(np.dot(mode, permu_mode.T)), norm_1 * norm_2)))
            measure_mode.append(np.add.reduce(np.diag(np.dot(mode, permu_mode.T)))/norm)

        return np.array(measure_mode), mesure_coor

