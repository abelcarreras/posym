from posym.operations import Operation
import numpy as np


class Identity(Operation):
    def __init__(self, coordinates, modes, symbols=None):
        super().__init__(coordinates, symbols)

        self._measure_mode = [1.0] * len(modes)
        self._measure_coor = [0.0] * len(modes)
