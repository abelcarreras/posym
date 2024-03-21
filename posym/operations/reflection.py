import numpy as np
from posym.operations import Operation
from posym.tools import standardize_vector


def cache_reflection(func):
    cache_dict = {}

    def wrapper_cache(rotation_axis):
        hash_key = tuple(rotation_axis)
        if hash_key in cache_dict:
            return cache_dict[hash_key]

        cache_dict[hash_key] = func(rotation_axis)
        return cache_dict[hash_key]

    return wrapper_cache


@cache_reflection
def reflection(reflection_axis):
    uax = np.dot(reflection_axis, reflection_axis)
    return np.identity(3) - 2*np.outer(reflection_axis, reflection_axis)/uax


class Reflection(Operation):
    def __init__(self, label, axis):
        super().__init__(label)
        self._order = 1
        self._determinant = -1

        self._axis = standardize_vector(axis)

    def __str__(self):
        axis_txt = '[{:8.3f} {:8.3f} {:8.3f}]'.format(*self._axis)
        return 'SymOp.Reflection {} {} <{}>'.format(self._label, axis_txt, hex(id(self)))

    def get_measure_modes(self, modes, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = reflection(rotated_axis)

        measure_mode = []
        for i, mode in enumerate(modes):

            operated_mode = np.dot(operation, np.array(mode).T).T
            norm = np.linalg.norm(mode)
            permu_mode = operated_mode[self.permutation]

            measure_mode.append(np.trace(np.dot(mode, permu_mode.T))/norm)

        return np.array(measure_mode)

    def get_measure_atom(self):

        measure_atoms = np.array([1 if i == p else 0 for i, p in enumerate(self.permutation)])

        return np.sum(measure_atoms)

    def get_measure_xyz(self, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = reflection(rotated_axis)

        measure_mode = []
        for axis in [[1, 0, 0], [0, 1 ,0], [0, 0, 1]]:
            operated_axis = np.dot(operation, axis)
            measure_mode.append(np.dot(axis, operated_axis))

        return np.sum(measure_mode)

    def get_displacements_projection(self, orientation=None):

        n_atoms = len(self.permutation)
        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = reflection(rotated_axis)

        cartesian_modes = np.identity(3 * n_atoms).reshape(3 * n_atoms, n_atoms, 3)

        projected_modes = []
        for i, mode in enumerate(cartesian_modes):
            operated_mode = np.dot(operation, np.array(mode).T).T
            projected_modes.append(operated_mode[self.permutation])

        return np.array(projected_modes)

    def get_measure_pos(self, coordinates, orientation=None, normalized=True):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = reflection(rotated_axis)

        operated_coor = np.dot(operation, coordinates.T).T
        mesure_coor = np.einsum('ij, ij -> ', coordinates, operated_coor[self.permutation])

        if normalized:
            mesure_coor /= np.einsum('ij, ij -> ', coordinates, coordinates)

        return mesure_coor

    def get_operated_coordinates(self, coordinates, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)
        operation = reflection(rotated_axis)

        #if self._permutation is None:
        #    from posym.operations import get_permutation_aprox
        #    self._permutation = get_permutation_aprox(operation, coordinates, symbols, self._order)

        operated_coor = np.dot(operation, coordinates.T).T
        return operated_coor[self.permutation]

    def get_overlap_func(self, op_function1, op_function2, orientation=None):

        rotated_axis = self._axis if orientation is None else orientation.apply(self._axis)

        operation = reflection(rotated_axis)

        fn_function_r = op_function1.copy()
        fn_function_r.apply_linear_transformation(operation)

        return (op_function2*fn_function_r).integrate

    def apply_rotation(self, orientation):
        self._axis = orientation.apply(self._axis)

    def inverse(self):
        return Reflection(self.label, axis=self._axis)

    @property
    def axis(self):
        return self._axis

    @property
    def matrix_representation(self):
        return reflection(self._axis)

if __name__ == '__main__':

    from posym.operations.rotation import rotation
    from posym.operations.inversion import inversion

    def d(a):
        # print('initial', np.array(a)/np.linalg.norm(a))
        return np.array(a)/np.linalg.norm(a)

    axis = d([0, 0, 1])
    matrix_1 = reflection(axis)
    print('m1\n', matrix_1)
    # exit()

    #matrix_2 = inversion()
    print('ini_axis: ', -d([0, 0, 1]))
    matrix_2 = rotation(2*np.pi/6*1, d([0, 0, 1]))
    print('m2\n', matrix_2)

    m12 = matrix_2 @ matrix_2 @ matrix_1 @ matrix_1

    m12 = m12
    print('m12')
    print(m12)

    from posym.operations.products import get_operation_from_matrix
    op = get_operation_from_matrix(m12)
    print('op: ', op)
    print(op.matrix_representation)

    print('check')
    print(np.round(op.matrix_representation - m12, decimals=3))

    exit()

    print('det', np.linalg.det(m12))
    eval, evec = np.linalg.eig(m12)
    print('eval', eval)
    print(evec.T)

    axis = np.real(evec.T[2])

    angle = np.arccos(np.real(eval[0]))
    print('angle: ', angle, 2*np.pi/3)

    exit()

    # m12_test = reflection(axis)
    m12_test = rotation(2*np.pi/3, axis)

    print('test')
    print(m12_test)
    print(m12_test - m12)




