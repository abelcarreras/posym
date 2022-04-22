__author__ = 'Abel Carreras'
__version__ = '0.1.2'

from posym.tools import list_round
from posym.pointgroup import PointGroup
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import numpy as np
import pandas as pd

cache_orientation = {}
def get_hash(coordinates, symbols, group):
    return hash((np.array2string(coordinates, precision=2),
                 tuple(symbols),
                 group))


class SymmetryBase():
    """
    This class is supposed to be used as a base for more complex symmetry objects

    """
    def __init__(self, group, rep, normalize=False):

        self._pg = PointGroup(group)
        self._group = group.lower()

        if isinstance(rep, str):
            if rep not in self._pg.ir_labels:
                raise Exception('Representation do not match with group\nAvailable: {}'.format(self._pg.ir_labels))
            self._op_representation = self._pg.ir_table[rep]
            #if normalize:
            #    self._op_representation /= self._pg.ir_table[rep]['E']

        elif isinstance(rep, pd.Series):
            if np.all(self._pg.ir_table.sort_index().index == rep.sort_index().index):
                self._op_representation = rep.reindex(self._pg.ir_table.index)
            else:
                raise Exception('Representation not in group')

        if normalize:
            op_rep = np.dot(self._pg.trans_matrix_norm, np.dot(self._pg.trans_matrix_inv, self._op_representation.values))
            self._op_representation = pd.Series(op_rep, index=self._pg.op_labels)

    def get_op_representation(self):
        return self._op_representation

    def get_ir_representation(self):
        ir_rep = np.dot(self._pg.trans_matrix_inv, self._op_representation.values)
        return pd.Series(ir_rep, index=self._pg.ir_labels)

    def get_point_group(self):
        return self._pg

    def __str__(self):

        ir_rep = self.get_ir_representation().values
        ir_rep = list_round(ir_rep)

        ir_labels = self.get_ir_representation().index

        str = ''
        for i, r in enumerate(ir_rep):
            #ir_norm = self._pg.ir_table[ir_labels[i]]['E']
            #r *= ir_norm
            #print('>> ', np.add.reduce(np.square(ir_rep[:i])), 'r:', r)
            if np.add.reduce(np.square(ir_rep[:i])) > 0 and r > 0:
                    str += ' + '
            elif r < 0:
                    str += ' - '
            if np.abs(r - 1) < 2e-2:
                str += ir_labels[i]
            elif np.abs(r) > 0:
                str += '{} {}'.format(abs(r), ir_labels[i])

        return str

    def __repr__(self):
        ir_rep = self.get_ir_representation().values
        ir_rep = list_round(ir_rep)

        ir_labels = self.get_ir_representation().index

        str = ''
        for i, r in enumerate(ir_rep):
            #print('>> ', np.add.reduce(ir_rep[:i]**2), 'r:', r)
            if np.add.reduce(np.square(ir_rep[:i])) > 0 and r > 0 and len(ir_rep[:i]) > 0:
                str += '+'
            if r == 1:
                str += ir_labels[i]
            elif r > 0:
                str += '{}{}'.format(r, ir_labels[i])
        return str

    def __add__(self, other):

        if self._group == other._group:
            return SymmetryBase(self._group,
                                self._op_representation + other._op_representation)

        raise Exception('Incompatible point groups')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):

        if isinstance(other, (float, int)):
            return SymmetryBase(self._group,
                                self._op_representation * other)

        elif isinstance(other, SymmetryBase):
            mul_rep = self._op_representation * other._op_representation

            return SymmetryBase(self._group, mul_rep)
        else:
            raise Exception('Symmetry operation not possible')


class SymmetryModes(SymmetryBase):
    def __init__(self, group, coordinates, modes, symbols, orientation_angles=None):

        pg = PointGroup(group)

        # set coordinates at geometrical center
        self._coordinates = np.array([c - np.average(coordinates, axis=0) for c in coordinates])

        self._modes = modes
        self._symbols = symbols

        self._coor_measures = []
        self._mode_measures = []

        if orientation_angles is None:
            self._angles = self.get_orientation(pg)
        else:
            self._angles = orientation_angles

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        operations_dic = pg.get_all_operations()

        for operation in pg.operations:
            mode_measures = []
            for op in operations_dic[operation.label]:
                mode_m = op.get_measure(self._coordinates, self._modes, self._symbols, orientation=rotmol)
                mode_measures.append(mode_m)

            mode_measures = np.average(mode_measures, axis=0)
            self._mode_measures.append(mode_measures)

        # De-normalization
        # self._mode_measures = np.dot(pg.trans_matrix, np.dot(pg.trans_matrix_inv_norm, self._mode_measures))

        total_state = pd.Series(np.add.reduce(self._mode_measures, axis=1), index=pg.op_labels)

        super().__init__(group, total_state)

    def get_state_mode(self, n):

        return SymmetryBase(group=self._group, rep=pd.Series(np.array(self._mode_measures).T[n],
                                                             index=self._pg.op_labels))

    def get_orientation(self, pg):

        hash_num = get_hash(self._coordinates, self._symbols, pg.label)
        if hash_num in cache_orientation:
            return cache_orientation[hash_num]

        def optimization_function(angles):

            rotmol = R.from_euler('zyx', angles, degrees=True)

            coor_measures = []
            for operation in pg.operations:
                coor_m = operation.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol)
                coor_measures.append(coor_m)

            # definition group measure
            return np.linalg.norm(coor_measures)

        # preliminar scan
        list_m = []
        list_a = []
        for i in np.arange(0, 180, 36):
            for j in np.arange(0, 180, 36):
                for k in np.arange(0, 180, 36):
                    list_m.append(optimization_function([i, j, k]))
                    list_a.append([i, j, k])

        initial = np.array(list_a[np.nanargmin(list_m)])
        res = minimize(optimization_function, initial, method='CG',
                       # bounds=((0, 360), (0, 360), (0, 360)),
                       # tol=1e-20
                       )
        cache_orientation[hash_num] = res.x
        return cache_orientation[hash_num]

    @property
    def get_measure_pos(self):
        return np.product(self._coor_measures)

    @property
    def opt_coordinates(self):
        rotmol = R.from_euler('zyx', self._angles, degrees=True)
        return rotmol.apply(self._coordinates)

    @property
    def orientation_angles(self):
        return self._angles


class SymmetryFunction(SymmetryBase):
    def __init__(self, group, function, orientation_angles=None, center=None):

        symbols, coordinates = function.get_environment_centers()

        if center is None:
            # center = function.global_center()
            center = np.average(coordinates, axis=0)
            function = function.copy()
            function.apply_translation(-np.array(center))
            coordinates = np.array([c - center for c in coordinates])

        pg = PointGroup(group)

        self._coordinates = np.array(coordinates)

        self._function = function
        self._symbols = symbols

        self._coor_measures = []
        self._operator_measures = []

        if orientation_angles is None:
            self._angles = self.get_orientation(pg)
        else:
            self._angles = orientation_angles

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._self_similarity = (self._function * self._function).integrate

        for operation in pg.operations:
            operations_dic = pg.get_all_operations()
            operator_measures = []
            for op in operations_dic[operation.label]:
                operator_m = op.get_measure_func(self._function, self._self_similarity, orientation=rotmol)
                operator_measures.append(operator_m)

            operator_measures = np.average(operator_measures, axis=0)
            self._operator_measures.append(operator_measures)

        # De-normalization
        #self._operator_measures = np.dot(pg.trans_matrix, np.dot(pg.trans_matrix_inv_norm, self._operator_measures))

        total_state = pd.Series(self._operator_measures, index=pg.op_labels)

        super().__init__(group, total_state)

    def get_orientation(self, pg):

        hash_num = get_hash(self._coordinates, self._symbols, pg.label)
        if hash_num in cache_orientation:
            return cache_orientation[hash_num]

        def optimization_function(angles):

            rotmol = R.from_euler('zyx', angles, degrees=True)

            coor_measures = []
            for operation in pg.operations:
                coor_m = operation.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol)
                coor_measures.append(coor_m)

            # definition group measure
            return np.linalg.norm(coor_measures)

        # preliminar scan
        list_m = []
        list_a = []
        for i in np.arange(0, 180, 36):
            for j in np.arange(0, 180, 36):
                for k in np.arange(0, 180, 36):
                    list_m.append(optimization_function([i, j, k]))
                    list_a.append([i, j, k])

        initial = np.array(list_a[np.nanargmin(list_m)])
        res = minimize(optimization_function, initial, method='CG',
                       # bounds=((0, 360), (0, 360), (0, 360)),
                       # tol=1e-20
                       )
        cache_orientation[hash_num] = res.x
        return cache_orientation[hash_num]

    @property
    def self_similarity(self):
        return self._self_similarity

    @property
    def get_measure_pos(self):
        return np.product(self._coor_measures)

    @property
    def opt_coordinates(self):
        rotmol = R.from_euler('zyx', self._angles, degrees=True)
        return rotmol.apply(self._coordinates)

    @property
    def orientation_angles(self):
        return self._angles


if __name__ == '__main__':

    from pyqchem import get_output_from_qchem, Structure
    from pyqchem.tools import get_geometry_from_pubchem

    coordinates = [[ 0.000000000+00,  0.000000000+00,  2.40297090e-01],
                   [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
                   [ 1.43261539e+00,  1.75444785e-16, -9.61188362e-01]]

    symbols = ['O', 'H', 'H']

    coordinates = [[0, 0, 0],
                   [ np.sqrt(8/9), 0, -1/3],
                   [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
                   [-np.sqrt(2/9), -np.sqrt(2/3), -1/3],
                   [0, 0, 1]]

    #coordinates = [[ 0.0000000000,    0.0000000000,    0.0000000000],
    #               [ 0.5541000000,    0.7996000000,    0.4965000000],
    #               [ 0.6833000000,   -0.8134000000,   -0.2536000000],
    #               [-0.7782000000,   -0.3735000000,    0.6692000000],
    #               [-0.4593000000,    0.3874000000,   -0.9121000000]]

    coordinates = [[-3.11301739e-06,  1.12541091e-05, -7.97835696e-06],
                   [-6.58614327e-02, -7.77865103e-01, -7.63860353e-01],
                   [-3.64119514e-02,  9.82136003e-01, -4.76386979e-01],
                   [-8.37587195e-01, -9.93456888e-02,  6.93899368e-01],
                   [ 9.39879258e-01, -1.04992740e-01,  5.46395830e-01]]

    symbols = ['C', 'H', 'H', 'H', 'H']


    if True:
        #mol = get_geometry_from_pubchem('methane')
        from pyqchem.file_io import read_structure_from_xyz

        # mol = get_geometry_from_pubchem('Buckminsterfullerene')
        mol = read_structure_from_xyz('../c60.xyz')

        coordinates = mol.get_coordinates()
        symbols = mol.get_symbols()

    from posym.operations.rotation import Rotation, rotation
    from posym.operations.reflection import Reflection, reflection
    from posym.operations.irotation import ImproperRotation
    from posym.operations import get_permutation_simple, get_cross_distance_table

    import matplotlib.pyplot as plt

    if True:
        def optimization_function(angles):
            rotmol = R.from_euler('zyx', angles, degrees=True)
            operation = Rotation(label='C2', axis=[0, 0, 1], order=2)
            # print(operation._axis, operation._order)
            coor_m = operation.get_measure_pos(np.array(coordinates), symbols, orientation=rotmol)
            operation = Reflection(label='s', axis=[0, 1, 0])
            # print(operation._axis, operation._order)
            coor_m2 = operation.get_measure_pos(np.array(coordinates), symbols, orientation=rotmol)
            operation = Rotation(label='C2', axis=[np.sqrt(2/9), 0, 1/3], order=2)
            coor_m3 = operation.get_measure_pos(np.array(coordinates), symbols, orientation=rotmol)
            operation = ImproperRotation(label='S4', axis=[np.sqrt(2/9), 0, 1/3], order=4)
            coor_m4 = operation.get_measure_pos(np.array(coordinates), symbols, orientation=rotmol)

            print(coor_m)
            #print(coor_m + coor_m2 + coor_m3 + coor_m4)
            return coor_m # + coor_m2 + coor_m3 + coor_m4
            #return np.product([coor_m, coor_m2, coor_m3, coor_m4])


        #exit()

        list_m = []
        list_a = []
        for i in np.arange(0, 180, 36):
            for j in np.arange(0, 180, 36):
                for k in np.arange(0, 180, 36):
                    list_m.append(optimization_function([i, j, k]))
                    list_a.append([i, j, k])

        print('------')
        initial = np.array(list_a[np.nanargmin(list_m)])
        #initial = [143.63674808,  -4.31338625,  70.70307201]
        # initial = [43.22189286,  43.86797247, 103.83908054]
        res = minimize(optimization_function, initial, method='CG',
                       #tol=1e20
                       )

        print('res', res.x)
        print('test', optimization_function(res.x))

        # res.x = [43.22189286,  43.86797247, 103.83908054]

        rotmol = R.from_euler('zyx', res.x, degrees=True)

    else:
        angles = [43.22189286,  43.86797247, 103.83908054]
        rotmol = R.from_euler('zyx', angles, degrees=True)


        #print(optimization_function(res.x))
        #rotmol = R.from_euler('zyx', res.x, degrees=True)


    #operation = Rotation(label='C3', axis=[0, 0, 1], order=3)
    #rotmol = R.from_euler('zyx', [0, 0, 0], degrees=True)
    #mode_m, coor_m = operation.get_measure(np.array(coordinates), modes, symbols, orientation=rotmol)
    #print('measure', coor_m)
    #rotmol = R.from_euler('zyx', [360./3, 0, 0], degrees=True)
    #print('---')

    print(Structure(coordinates, symbols))

    # rotations
    rotated_axis = rotmol.apply([0, 0, 1])
    operation = rotation(2*np.pi/2, rotated_axis)
    print('axis', rotated_axis)

    # reflection
    #rotated_axis_r = rotmol.apply([0, 1,  0])
    #operation = reflection(rotated_axis_r)
    #print('axis_r', rotated_axis_r)

    # C2 rotation
    # rotated_axis_i = rotmol.apply([(np.sqrt(8 / 9))/2, 0, (1-1 / 3)/2])

    # S4 rotation
    #rotated_axis_i = rotmol.apply([np.sqrt(2 / 9), 0, 1 / 3])
    #operation1 = rotation(2*np.pi/4, rotated_axis_i)
    #operation2 = reflection(rotated_axis_i)
    #operation = np.dot(operation2, operation1)
    #print('axis_i', rotated_axis_i)

    #coordinates = rotmol.apply(coordinates)
    #print(Structure(coordinates, symbols))

    print('det:', np.linalg.det(operation))
    permu_coor = np.dot(operation, np.array(coordinates).T).T

    print(Structure(permu_coor, symbols))


    distance_table = get_cross_distance_table(coordinates, permu_coor)
    perm = get_permutation_simple(distance_table, symbols)
    #print(np.round(distance_table[perm], 3))
    print(perm)

    permu_coor = permu_coor[perm]

    print(Structure(permu_coor, symbols))

    print('****')
    print(permu_coor)
    a = np.average(np.linalg.norm(np.subtract(coordinates, permu_coor), axis=0))
    print(a)
