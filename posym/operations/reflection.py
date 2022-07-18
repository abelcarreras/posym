import numpy as np
from posym.operations import Operation
from posym.tools import standardize_vector


def reflection(reflection_axis):
    uax = np.dot(reflection_axis, reflection_axis)

    return np.identity(3) - 2*np.outer(reflection_axis, reflection_axis)/uax


class Reflection(Operation):
    def __init__(self, label, axis):
        super().__init__(label)

        self._axis = standardize_vector(axis)

    def __hash__(self):
        axis = np.round(self._axis, decimals=6)
        axis[axis == 0.] = 0.

        return hash((self._label,
                     np.array2string(axis)
                     ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_measure(self, coordinates, modes, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = reflection(rotated_axis)

        measure_mode = []
        mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

        for i, mode in enumerate(modes):

            operated_mode = np.dot(operation, np.array(mode).T).T
            norm = np.linalg.norm(mode)
            permu_mode = operated_mode[permu]

            measure_mode.append(np.trace(np.dot(mode, permu_mode.T))/norm)

        return np.array(measure_mode)

    def get_measure_pos(self, coordinates, symbols, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        operation = reflection(rotated_axis)
        mesure_coor, permu = self.get_permutation(operation, coordinates, symbols)

        return mesure_coor

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        operation = reflection(rotated_axis)

        fn_function_r = op_function1.copy()
        fn_function_r.apply_linear_transformation(operation)

        return (op_function2*fn_function_r).integrate


    @property
    def axis(self):
        return self._axis

    @property
    def operation_matrix_list(self):
        return [reflection(self._axis)]

    def __mul__(self, other):
        if not other.__class__.__bases__[0] is Operation:
            raise Exception('Product only defined between Operation subclasses')
        else:
            op_list = []
            for op_mat in other.operation_matrix_list:
                new_axis = np.dot(op_mat, self._axis)
                op_list.append(Reflection(self._label, new_axis))

            return op_list
