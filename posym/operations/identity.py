from posym.operations import Operation
import numpy as np


class Identity(Operation):
    def __init__(self, label):
        super().__init__(label)

    def get_measure(self, coordinates, modes, symbols, orientation=None):
        self._measure_mode = [1.0] * len(modes)
        self._measure_coor = 0.0

        return np.array(self._measure_mode), self._measure_coor

