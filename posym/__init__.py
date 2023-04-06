__author__ = 'Abel Carreras'
__version__ = '0.5.0'

from posym.tools import list_round
from posym.pointgroup import PointGroup
from posym.basis import BasisFunction
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, basinhopping
import numpy as np
import pandas as pd
import itertools

cache_orientation = {}
def get_hash(coordinates, symbols, group):
    return hash((np.array2string(coordinates, precision=2),
                 tuple(symbols),
                 group))


def get_simple(rep, group):
    return pd.Series(rep.values, index=rep.index)


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

    def get_reduced_op_representation(self):
        red_values = []
        for value in self._op_representation.values:
            red_values.append(np.average(value))
        return pd.Series(red_values, index=self._op_representation.index)

    def get_op_representation(self):
        return self._op_representation

    def get_ir_representation(self):
        ir_rep = np.dot(self._pg.trans_matrix_inv, self.get_reduced_op_representation().values)
        return pd.Series(ir_rep, index=self._pg.ir_labels)

    def get_point_group(self):
        return self._pg

    def __str__(self):

        ir_rep = self.get_ir_representation().values
        ir_rep = list_round(ir_rep)

        ir_labels = self.get_ir_representation().index

        str = ''
        for i, r in enumerate(ir_rep):
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

        txt = ''
        for i, r in enumerate(ir_rep):
            # print('>> ', np.add.reduce(ir_rep[:i]**2), 'r:', r)
            if np.add.reduce(np.square(ir_rep[:i])) > 0 and r > 0 and len(ir_rep[:i]) > 0:
                txt += '+'
            if r == 1:
                txt += ir_labels[i]
            elif r > 0:
                txt += '{}{}'.format(r, ir_labels[i])
        return txt

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


class SymmetryMoleculeBase(SymmetryBase):
    def __init__(self, group, coordinates, symbols, total_state=None, orientation_angles=None, center=None,
                 fast_optimization=True):

        self._setup_structure(coordinates, symbols, group, center, orientation_angles, fast_optimization=fast_optimization)

        if total_state is None:
            rotmol = R.from_euler('zyx', self._angles, degrees=True)
            geom_center = np.average(coordinates, axis=0)
            centered_coor = np.subtract(coordinates, geom_center)

            self._operator_measures = []
            for operation in self._pg.operations:
                operator_measures = []
                for op in self._pg.get_sub_operations(operation.label):
                    overlap = op.get_measure_pos(centered_coor, symbols, orientation=rotmol)
                    operator_measures.append(overlap)

                self._operator_measures.append(np.array(operator_measures))

            total_state = pd.Series(self._operator_measures, index=self._pg.op_labels)

        super().__init__(group, total_state)

    def _setup_structure(self, coordinates, symbols, group, center, orientation_angles, fast_optimization=True):

        self._coordinates = np.array(coordinates)
        self._symbols = symbols
        self._pg = PointGroup(group)

        if '_center' not in self.__dir__():
            self._center = center
            if self._center is None:
                self._center = np.average(self._coordinates, axis=0)

            self._coordinates = np.array([c - self._center for c in self._coordinates])

        if orientation_angles is None:
            self._angles = self.get_orientation(fast_optimization=fast_optimization)
        else:
            self._angles = orientation_angles

    def get_orientation(self, fast_optimization=True):
        """
        get orientation angles for optimum orientation.
        Use full=False to orient perfect symmetric molecules. Use full=True to orient quasi symmetric molecules

        :param fast_optimization: if True use only a subset of symmetry elements (for exact symmetry objets)
        :return:
        """

        hash_num = get_hash(self._coordinates, self._symbols, self._pg.label)
        if hash_num in cache_orientation:
            return cache_orientation[hash_num]

        def optimization_function_simple(angles):

            rotmol = R.from_euler('zyx', angles, degrees=True)

            coor_measures = []
            for operation in self._pg.operations:
                coor_m = operation.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol, normalized=False)
                coor_measures.append(coor_m)

            # definition group measure
            return -np.sum(coor_measures)

        def optimization_function_full(angles):

            rotmol = R.from_euler('zyx', angles, degrees=True)

            coor_measures = []
            for operation in self._pg.operations:
                for sub_operation in self._pg.get_sub_operations(operation.label):
                    coor_m = sub_operation.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol, normalized=False)
                    coor_measures.append(coor_m)

            # definition group measure
            return -np.sum(coor_measures)

        optimization_function = optimization_function_simple if fast_optimization else optimization_function_full

        # preliminary scan
        guess_angles = ref_value = None
        for angles in itertools.product(np.arange(0, 180, 10), np.arange(0, 180, 10), np.arange(0, 180, 10)):
            value = optimization_function(angles)
            if ref_value is None or value < ref_value:
                ref_value = value
                guess_angles = angles

        result = minimize(optimization_function, guess_angles, method='CG',)

        cache_orientation[hash_num] = result.x
        return cache_orientation[hash_num]

    def get_oriented_operations(self):
        import copy
        rotmol = R.from_euler('zyx', self.orientation_angles, degrees=True)

        operations_list = []
        for operation in self._pg.operations:
            for sub_operation in self._pg.get_sub_operations(operation.label):
                sub_operation = copy.deepcopy(sub_operation)
                sub_operation.apply_rotation(rotmol)
                operations_list.append(sub_operation)

        return operations_list

    @property
    def measure_pos(self):

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        coor_measures = []
        for operation in self._pg.operations:
            for sub_operation in self._pg.get_sub_operations(operation.label):
                coor_m = sub_operation.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol)
                coor_measures.append(coor_m)

        return 1-np.average(coor_measures)

    @property
    def opt_coordinates(self):
        rotmol = R.from_euler('zyx', self._angles, degrees=True)
        return rotmol.apply(self._coordinates)

    @property
    def orientation_angles(self):
        return self._angles

    @property
    def center(self):
        return self._center


class SymmetryModes(SymmetryMoleculeBase):
    def __init__(self, group, coordinates, modes, symbols, orientation_angles=None):

        self._setup_structure(coordinates, symbols, group, None, orientation_angles)

        self._modes = modes

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._mode_measures = []
        for operation in self._pg.operations:
            mode_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                mode_m = op.get_measure_modes(self._coordinates, self._modes, self._symbols, orientation=rotmol)
                mode_measures.append(mode_m)

            mode_measures = np.array(mode_measures)
            self._mode_measures.append(mode_measures)

        mode_measures_total = []
        for op in self._mode_measures:
            op_list = []
            for m in op:
                op_list.append(sum(m))
            mode_measures_total.append(op_list)

        # reshape mode measures
        reshaped_modes_measures = []
        for m in range(len(self._mode_measures[0].T)):
            reshaped_modes_measures.append([k[:, m] for k in self._mode_measures])

        self._mode_measures = reshaped_modes_measures

        total_state = pd.Series(mode_measures_total, index=self._pg.op_labels)

        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0,0,0])

    def get_state_mode(self, n):
        return SymmetryBase(group=self._group, rep=pd.Series(self._mode_measures[n],
                                                             index=self._pg.op_labels))


class SymmetryModesFull(SymmetryMoleculeBase):
    def __init__(self, group, coordinates, symbols, orientation_angles=None):

        self._setup_structure(coordinates, symbols, group, None, orientation_angles)

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        trans_rots = []
        for label in self._pg.ir_table.rotations + self._pg.ir_table.translations:
            trans_rots.append(self._pg.ir_table[label].values)

        trans_rots = np.sum(trans_rots, axis=0)

        self._mode_measures = []
        for operation in self._pg.operations:
            mode_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                measure_xyz = op.get_measure_xyz(orientation=rotmol)
                measure_atom = op.get_measure_atom(self._coordinates, self._symbols, orientation=rotmol)

                mode_measures.append(measure_xyz * measure_atom)

            mode_measures = np.array(mode_measures)
            self._mode_measures.append(mode_measures)

        self._mode_measures = np.array(self._mode_measures, dtype=object).flatten() - trans_rots
        total_state = pd.Series(self._mode_measures, index=self._pg.op_labels)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0,0,0])


class SymmetryFunction(SymmetryMoleculeBase):
    def __init__(self, group, function, orientation_angles=None, center=None):

        symbols, coordinates = function.get_environment_centers()

        self._setup_structure(coordinates, symbols, group, center, orientation_angles)

        self._function = function.copy()
        self._function.apply_translation(-np.array(self._center))

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._self_similarity = (self._function * self._function).integrate

        self._operator_measures = []
        for operation in self._pg.operations:
            operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                overlap = op.get_overlap_func(self._function, self._function, orientation=rotmol)
                operator_measures.append(overlap/self._self_similarity)

            self._operator_measures.append(np.array(operator_measures))

        total_state = pd.Series(self._operator_measures, index=self._pg.op_labels)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0,0,0])

    @property
    def self_similarity(self):
        return self._self_similarity


class SymmetryWaveFunction(SymmetryMoleculeBase):
    def __init__(self, group, alpha_orbitals, beta_orbitals, center=None, orientation_angles=None):

        # generate copy to not modify original orbitals
        alpha_orbitals = [f.copy() for f in alpha_orbitals]
        beta_orbitals = [f.copy() for f in beta_orbitals]

        function = BasisFunction([], [])
        for f in alpha_orbitals:
            function = function + f
        for f in beta_orbitals:
            function = function + f

        symbols, coordinates = function.get_environment_centers()

        self._setup_structure(coordinates, symbols, group, center, orientation_angles)

        data_set = set(list(alpha_orbitals) + list(beta_orbitals))
        for f in data_set:
            f.apply_translation(-np.array(self._center))

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        def get_overlaps(orbitals):

            operator_overlaps_total = []
            for i, a_orb in enumerate(orbitals):
                overlap_row = []
                for j, b_orb in enumerate(orbitals):
                    self_similarity = (a_orb*a_orb).integrate * (b_orb*b_orb).integrate

                    overlaps_list = []
                    for operation in self._pg.operations:
                        operator_overlaps = []
                        for op in self._pg.get_sub_operations(operation.label):
                            overlap = op.get_overlap_func(a_orb, b_orb, orientation=rotmol)
                            operator_overlaps.append(overlap/self_similarity)

                        operator_overlaps = np.array(operator_overlaps)
                        # operator_overlaps = np.average(operator_overlaps, axis=0)
                        overlaps_list.append(operator_overlaps)

                    overlap_row.append(overlaps_list)
                operator_overlaps_total.append(overlap_row)

            operator_overlaps = []
            for k in range(self._pg.n_ir):
                multi_over = []
                n_degenerate = len(operator_overlaps_total[0][0][k])
                for m in range(n_degenerate):
                    matrix = np.zeros((len(orbitals), len(orbitals)))
                    for i in range(len(orbitals)):
                        for j in range(len(orbitals)):
                            matrix[i, j] = operator_overlaps_total[i][j][k][m]

                    multi_over.append(np.linalg.det(matrix))
                operator_overlaps.append(np.array(multi_over, dtype=float))

            return operator_overlaps

        if len(alpha_orbitals) > 0 and len(beta_orbitals) > 0:

            operator_overlaps_alpha = get_overlaps(alpha_orbitals)
            operator_overlaps_beta = get_overlaps(beta_orbitals)

            total_state = pd.Series(operator_overlaps_alpha, index=self._pg.op_labels) * \
                          pd.Series(operator_overlaps_beta, index=self._pg.op_labels)

            super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])

        elif len(alpha_orbitals) > 0:

            operator_overlaps_alpha = get_overlaps(alpha_orbitals)
            total_state = pd.Series(operator_overlaps_alpha, index=self._pg.op_labels)
            super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])

        elif len(beta_orbitals) > 0:

            operator_overlaps_beta = get_overlaps(beta_orbitals)
            total_state = pd.Series(operator_overlaps_beta, index=self._pg.op_labels)
            super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])


class SymmetryWaveFunctionCI(SymmetryMoleculeBase):
    def __init__(self, group, orbitals,
                 configurations,
                 center=None, orientation_angles=None):

        # generate copy to not modify original orbitals
        orbitals = [f.copy() for f in orbitals]

        function = BasisFunction([], [])
        for f in orbitals:
            function = function + f

        symbols, coordinates = function.get_environment_centers()
        self._setup_structure(coordinates, symbols, group, center, orientation_angles)

        for f in orbitals:
            f.apply_translation(-np.array(self._center))

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        def set_linear_combination(overlaps_matrix, configurations):

            def get_sub_matrix(occupations1, occupations2):
                cmat = overlaps_matrix.copy()
                n = len(occupations1)
                for i, (c1, c2) in enumerate(zip(occupations1[::-1], occupations2[::-1])):
                    if c1 == 0:
                        cmat = np.delete(cmat, n - i - 1, 0)
                    if c2 == 0:
                        cmat = np.delete(cmat, n - i - 1, 1)
                return cmat

            overlap = 0
            for i, conf1 in enumerate(configurations):
                for j, conf2 in enumerate(configurations):
                    new_matrix_alpha = get_sub_matrix(conf1['occupations']['alpha'], conf2['occupations']['alpha'])
                    overlap_alpha = np.linalg.det(new_matrix_alpha)

                    new_matrix_beta = get_sub_matrix(conf1['occupations']['beta'], conf2['occupations']['beta'])
                    overlap_beta = np.linalg.det(new_matrix_beta)

                    overlap += overlap_alpha * overlap_beta * conf1['amplitude']*conf2['amplitude']
            return overlap

        def get_overlaps(orbitals, configurations):

            operator_overlaps_total = []
            for i, a_orb in enumerate(orbitals):
                overlap_row = []
                for j, b_orb in enumerate(orbitals):
                    self_similarity = (a_orb*a_orb).integrate * (b_orb*b_orb).integrate

                    overlaps_list = []
                    for operation in self._pg.operations:
                        operator_overlaps = []
                        for op in self._pg.get_sub_operations(operation.label):
                            overlap = op.get_overlap_func(a_orb, b_orb, orientation=rotmol)
                            operator_overlaps.append(overlap/self_similarity)

                        operator_overlaps = np.array(operator_overlaps)
                        # operator_overlaps = np.average(operator_overlaps, axis=0)
                        overlaps_list.append(operator_overlaps)

                    overlap_row.append(overlaps_list)
                operator_overlaps_total.append(overlap_row)

            operator_overlaps = []
            for k in range(self._pg.n_ir):
                multi_over = []
                n_degenerate = len(operator_overlaps_total[0][0][k])
                for m in range(n_degenerate):
                    matrix = np.zeros((len(orbitals), len(orbitals)))
                    for i in range(len(orbitals)):
                        for j in range(len(orbitals)):
                            matrix[i, j] = operator_overlaps_total[i][j][k][m]

                    multi_over.append(set_linear_combination(matrix, configurations))
                operator_overlaps.append(np.array(multi_over, dtype=float))

            return operator_overlaps

        operator_overlaps = get_overlaps(orbitals, configurations)
        total_state = pd.Series(operator_overlaps, index=self._pg.op_labels)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, center=[0, 0, 0])


if __name__ == '__main__':

    from pyqchem import get_output_from_qchem, Structure
    from pyqchem.tools import get_geometry_from_pubchem


    coordinates = [[ 0.000000000+00,  0.000000000+00,  2.40297090e-01],
                   [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
                   [ 1.43261539e+00,  1.75444785e-16, -9.61188362e-01]]

    symbols = ['O', 'H', 'H']

    coordinates_ = [[0, 0, 0],
                   [ np.sqrt(8/9), 0, -1/3],
                   [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
                   [-np.sqrt(2/9), -np.sqrt(2/3), -1/3],
                   [0, 0, 1]]

    coordinates_ = [[-3.11301739e-06,  1.12541091e-05, -7.97835696e-06],
                   [-6.58614327e-02, -7.77865103e-01, -7.63860353e-01],
                   [-3.64119514e-02,  9.82136003e-01, -4.76386979e-01],
                   [-8.37587195e-01, -9.93456888e-02,  6.93899368e-01],
                   [ 9.39879258e-01, -1.04992740e-01,  5.46395830e-01]]

    symbols_ = ['C', 'H', 'H', 'H', 'H']

    sm = SymmetryModesFull('c2v', coordinates, symbols)
    print(sm.get_point_group())
    print(sm)
    mb = SymmetryMoleculeBase('c2v', coordinates, symbols)
    from posym.algebra import norm
    print('Coor measure: ', mb, '(', norm(mb), ')')

    exit()


    if False:
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

    coordinates = np.array(coordinates)
    operation = Rotation(label='C2', axis=[0, 0, 1], order=2)
    operation = Reflection(label='C2', axis=[0, 0, 1])

    a = operation.get_measure_xyz(coordinates, symbols)
    print(a)

    exit()

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

        print(np.sort(list_m))
        exit()
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
