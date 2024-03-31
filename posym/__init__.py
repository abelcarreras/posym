__author__ = 'Abel Carreras'
__version__ = '1.1'

from posym.tools import list_round, get_principal_axis_angles
from posym.pointgroup import PointGroup
from posym.basis import BasisFunction
from posym.config import Configuration
from posym.tools import uniform_euler_scan, collapse_limit
from posym.permutation import generate_permutation_set
from posym.permutation.hungarian import get_permutation_hungarian
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import warnings


cache_orientation = {}
def get_hash(coordinates, symbols, group):
    return hash((np.array2string(coordinates, precision=2), tuple(symbols), group))


class SymmetryObject:
    """
    Main symmetry object that abstracts an element in the g-module space

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
            return SymmetryObject(self._group,
                                  self._op_representation + other._op_representation)

        raise Exception('Incompatible point groups')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):

        if isinstance(other, (float, int)):
            return SymmetryObject(self._group,
                                  self._op_representation * other)

        elif isinstance(other, SymmetryObject):
            mul_rep = self._op_representation * other._op_representation

            return SymmetryObject(self._group, mul_rep)
        else:
            raise Exception('Symmetry operation not possible')


class SymmetryMolecule(SymmetryObject):
    """
    Symmetry of molecular geometry
    """
    def __init__(self, group, coordinates, symbols, total_state=None, orientation_angles=None, center=None, permutation_set=None):
        """

        :param group: symmetry point group
        :param coordinates: molecular coordinates
        :param symbols: atomic symbols
        :param total_state: symmetry operations overlaps (SOEV's) as a panda Series object
        :param orientation_angles: orientation angles
        :param center: center of symmetry group [x, y, z]
        """

        self._setup_structure(coordinates, symbols, group, center, orientation_angles, permutation_set=permutation_set)

        if total_state is None:
            # m = self.measure_pos
            self._generate_permutation_set(self._angles)
            rotmol = R.from_euler('zyx', self._angles, degrees=True)
            self._operator_measures = []
            for operation in self._pg.operations:
                operator_measures = []
                for op in self._pg.get_sub_operations(operation.label):

                    # check if all atoms are collapsed in a point
                    if collapse_limit(self._coordinates):
                        operator_measures.append(1)
                        continue

                    overlap = op.get_measure_pos(self._coordinates, orientation=rotmol)
                    operator_measures.append(overlap)

                # print('operator measures', operator_measures)
                self._operator_measures.append(np.array(operator_measures))

            total_state = pd.Series(self._operator_measures, index=self._pg.op_labels)

        if not self.check_permutation_coherence and not collapse_limit(self.symmetrized_coordinates):
            warnings.warn('Incoherence found in symmetrized structure. Symmetry measure may be incorrect')

        super().__init__(group, total_state)

    def _setup_structure(self, coordinates, symbols, group, center, orientation_angles, permutation_set=None):

        conf = Configuration()

        self._coordinates = np.array(coordinates)
        self._symbols = symbols
        self._pg = PointGroup(group)
        self._permutation_set = {}

        if '_center' not in self.__dir__():
            self._center = center
            if self._center is None:
                self._center = np.average(self._coordinates, axis=0)

            self._coordinates = np.array([c - self._center for c in self._coordinates])

        if orientation_angles is None:
            self._angles = self.get_orientation(fast_optimization=conf.fast_optimization, scan_step=conf.scan_steps)
        else:
            self._angles = orientation_angles

        # manual permutation
        if permutation_set is not None:
            self._permutation_set[tuple(orientation_angles)] = {gen: perm for gen, perm in
                                                                zip(self._pg.generators,permutation_set)}

        self._generate_permutation_set(self._angles)

    def get_orientation(self, fast_optimization=True, scan_step=20, guess_angles=None):
        """
        get orientation angles for optimum orientation.
        Use full=False to orient perfect symmetric molecules. Use full=True to orient quasi symmetric molecules

        :param fast_optimization: if True use only a subset of symmetry elements (for exact symmetry objets)
        :param scan_step: step angle (deg) use for the preliminary scan
        :return:
        """

        hash_num = get_hash(self._coordinates, self._symbols, self._pg.label)
        if hash_num in cache_orientation:
            return cache_orientation[hash_num]

        # optimization functions
        def optimization_function_simple(angles):
            """
            This function uses only one operation of each type (described in the IR table).
            This approach works well when the molecule has a symmetry close to the group
            """
            self._generate_permutation_set(angles)
            rotmol = R.from_euler('zyx', angles, degrees=True)

            coor_measures = []
            for operation in self._pg.operations:
                coor_m = operation.get_measure_pos(self._coordinates, orientation=rotmol, normalized=False)
                coor_measures.append(coor_m)

            # get most symmetric IR value
            return -np.dot(coor_measures, self._pg.trans_matrix_inv[0])

        def optimization_function_full(angles):
            """
            This function uses all operations of the group and averages the overlap of equivalent operations
            """

            self._generate_permutation_set(angles)
            rotmol = R.from_euler('zyx', angles, degrees=True)

            operator_measures = []
            for operation in self._pg.operations:
                sub_operator_measures = []
                for op in self._pg.get_sub_operations(operation.label):
                    overlap = op.get_measure_pos(self._coordinates, orientation=rotmol)
                    sub_operator_measures.append(overlap)
                operator_measures.append(np.average(sub_operator_measures))

            # get most symmetric IR value
            return -np.dot(operator_measures, self._pg.trans_matrix_inv[0])

        # define if use simple function (faster) or full (slower)
        optimization_function = optimization_function_simple if fast_optimization else optimization_function_full

        # define better orientation for preliminar scan
        pai_angles = get_principal_axis_angles(self._coordinates)

        # preliminary scan
        if guess_angles is None:
            guess_angles = ref_value = None
            for angles in uniform_euler_scan(90, scan_step):
                angles -= pai_angles
                value = optimization_function(angles)
                if ref_value is None or value < ref_value:
                    ref_value = value
                    guess_angles = angles

        result = minimize(optimization_function, guess_angles, method='CG')

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

    def print_operations_info(self):
        from posym.permutation import Permutation

        print('\nOperations list (molecule orientation)'
              '\n--------------------------------------')
        self._generate_permutation_set(self._angles)
        for op in self.get_oriented_operations():
            # print('Operation object: ', op)
            print('label:', op.label)
            try:
                print('Order:', op.order)
                print('Exponent:', op.exp)
            except AttributeError:
                pass
            try:
                print('Axis:', op.axis)
            except AttributeError:
                pass

            print('permutation:', op.permutation)
            permutation = Permutation(op.permutation)
            print('orbits: ', permutation.get_orbits())

            print('Matrix representation:')
            print(np.round(op.matrix_representation, decimals=6))
            print()

    def _generate_permutation_set(self, angles, force_reset=False):

        rotmol = R.from_euler('zyx', angles, degrees=True)
        dict_key = tuple(angles)

        if dict_key not in self._permutation_set or force_reset:

            if collapse_limit(self._coordinates):
                self._permutation_set[dict_key] = next(generate_permutation_set(self._pg.generators, self._symbols))
                return

            # Hungarian algorithm (approximated)
            if Configuration().algorithm == 'hungarian':
                permutation_set = {}
                for gen in self._pg.generators:
                    rot_coor = rotmol.inv().apply(self._coordinates)
                    permutation_set[gen] = get_permutation_hungarian(gen.matrix_representation, rot_coor, self._symbols)
                self._permutation_set[dict_key] = permutation_set

            # Brute force algorithm (exact)
            elif Configuration().algorithm == 'exact':
                ir_rep_diff_max = -100

                class NotValidPermutation(Exception): pass

                for permutation_set in generate_permutation_set(self._pg.generators, self._symbols):

                    try:
                        operator_measures = []
                        for operation in self._pg.operations:
                            sub_operator_measures = []
                            for op in self._pg.get_sub_operations(operation.label):
                                if op.set_permutation_set(permutation_set, self._symbols) is None:
                                    raise NotValidPermutation

                                overlap = op.get_measure_pos(self._coordinates, orientation=rotmol)
                                sub_operator_measures.append(overlap)

                            operator_measures.append(np.average(sub_operator_measures))

                        ir_rep_diff = np.dot(operator_measures, self._pg.trans_matrix_inv[0])

                        if ir_rep_diff_max < ir_rep_diff:
                            ir_rep_diff_max = ir_rep_diff
                            self._permutation_set[dict_key] = permutation_set

                    except NotValidPermutation:
                        continue
            else:
                raise Exception('Permutation algorithm not recognized ')

        for operation in self._pg.operations:
            operation.set_permutation_set(self._permutation_set[dict_key], self._symbols, ignore_compatibility=True)
            for op in self._pg.get_sub_operations(operation.label):
                op.set_permutation_set(self._permutation_set[dict_key], self._symbols, ignore_compatibility=True)

    @property
    def measure(self):
        norm = self.get_reduced_op_representation().values[0]
        return 100*(1-np.array(self.get_ir_representation().values[0])/norm)

    @property
    def measure_pos(self):

        if collapse_limit(self._coordinates):
            return 0.0

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._generate_permutation_set(self._angles)

        operator_measures = []
        for operation in self._pg.operations:
            sub_operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                overlap = op.get_measure_pos(self._coordinates, orientation=rotmol)
                sub_operator_measures.append(overlap)

            operator_measures.append(np.average(sub_operator_measures))

        # get most symmetric IR value
        ir_rep_diff = np.dot(operator_measures, self._pg.trans_matrix_inv[0])

        # return csm
        return 100 * (1 - ir_rep_diff)

    @property
    def check_permutation_coherence(self, tolerance=0.95):
        """
        check coherence of the permutation by checking the symmetry measure of
        the symmetrized structure.  This function only checks the symmetrized structure.
        The measure may still be correct.

        :param tolerance: tolenrece value for the final overlap measure of the structure
        :return: True or False
        """

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        operator_measures = []
        for operation in self._pg.operations:
            sub_operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                if collapse_limit(self._coordinates):
                    sub_operator_measures.append(1)
                    continue

                overlap = op.get_measure_pos(self.symmetrized_coordinates, orientation=rotmol)
                sub_operator_measures.append(overlap)

            operator_measures.append(np.average(sub_operator_measures))

        # get most symmetric IR value
        ir_rep_diff = np.dot(operator_measures, self._pg.trans_matrix_inv[0])

        return ir_rep_diff > tolerance

    @property
    def opt_coordinates(self):
        rotmol = R.from_euler('zyx', self._angles, degrees=True)
        return rotmol.apply(self._coordinates)

    @property
    def symmetrized_coordinates(self):

        self._generate_permutation_set(self._angles)
        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        structure_list = []
        for op in self._pg.all_operations:
            structure = op.get_operated_coordinates(self._coordinates, orientation=rotmol)
            structure_list.append(structure)

        return np.average(structure_list, axis=0)

    @property
    def orientation_angles(self):
        return self._angles

    @property
    def center(self):
        return self._center


class SymmetryNormalModes(SymmetryMolecule):
    """
    get symmetry of the normal modes.
    """
    def __init__(self, group, coordinates, modes, symbols, orientation_angles=None, center=None):
        """

        :param group: symmetry point group
        :param coordinates: atomic coordinates
        :param modes: list of normal modes separated by atoms [[[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], ...]
        :param symbols: atomic symbols
        :param orientation_angles: list of 3 Euler angles [pitch, yaw, roll]
        :param center: center of symmetry group [x, y, z]
        """

        self._setup_structure(coordinates, symbols, group, center, orientation_angles)

        self._modes = modes

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._mode_measures = []
        for operation in self._pg.operations:
            mode_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                mode_m = op.get_measure_modes(self._modes, orientation=rotmol)
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

        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])

    def get_state_mode(self, n):
        return SymmetryObject(group=self._group, rep=pd.Series(self._mode_measures[n],
                                                               index=self._pg.op_labels))

    def get_number_of_modes(self):
        return len(self._mode_measures)


class SymmetryAtomDisplacements(SymmetryMolecule):
    """
    get symmetry of atom displacements . This is equivalent to the sum of the symmetries of all
    normal modes (gamma 3N).
    """
    def __init__(self, group, coordinates, symbols, orientation_angles=None):
        """

        :param group: symmetry group
        :param coordinates: atomic coordinates
        :param symbols: atomic symbols
        :param orientation_angles: list of 3 Euler angles [pitch, yaw, roll]
        """

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
                measure_atom = op.get_measure_atom()
                mode_measures.append(measure_xyz * measure_atom)

            mode_measures = np.array(mode_measures)
            self._mode_measures.append(mode_measures)

        self._mode_measures = np.array(self._mode_measures, dtype=object).flatten() - trans_rots
        total_state = pd.Series(self._mode_measures, index=self._pg.op_labels)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])


class SymmetryAdaptedCoordinates(SymmetryMolecule):
    """
    get symmetry of atom displacements . This is equivalent to the sum of the symmetries of all
    normal modes (gamma 3N).
    """
    def __init__(self, group, coordinates, symbols, orientation_angles=None):
        """

        :param group: symmetry group
        :param coordinates: atomic coordinates
        :param symbols: atomic symbols
        :param orientation_angles: list of 3 Euler angles [pitch, yaw, roll]
        """

        self._setup_structure(coordinates, symbols, group, None, orientation_angles)

        if not self._pg.is_abelian:
            raise Exception('Not implemented for non-Abelian groups')

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._modes = []
        self._op_overlaps = []
        for traces_vector in self._pg.trans_matrix.T:
            projection = np.zeros((3 * len(coordinates), len(coordinates), 3))
            for operation, trace in zip(self._pg.operations, traces_vector):
                projection += trace * operation.get_displacements_projection(orientation=rotmol)

            def linear_indepedent(vect):
                for m in self._modes:
                    dot = np.abs(np.abs(np.dot(m.flatten(), vect.flatten())) - 1)
                    if dot < 1e-5:
                        return False
                return True

            for m in projection:
                norm = np.linalg.norm(m.flatten())
                if abs(norm) > 1e-5 and linear_indepedent(m/norm):
                    self._modes.append(m/norm)
                    self._op_overlaps.append(traces_vector)

        total_state = pd.Series(np.sum(self._op_overlaps, axis=0), index=self._pg.op_labels)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])

    def get_symmetry_adapted_coordinates(self):
        return np.array(self._modes).tolist()

    def get_state_mode(self, n):
        return SymmetryObject(group=self._group, rep=pd.Series(self._op_overlaps[n],
                                                               index=self._pg.op_labels))

    def get_number_of_modes(self):
        return len(self._op_overlaps)


class SymmetryGaussianLinear(SymmetryMolecule):
    """
    get symmetry from a function defined in the basis of Gaussian functions (BasisFunction object)

    """
    def __init__(self, group, function, orientation_angles=None, center=None):
        """

        :param group: symmetry group
        :param function: the function (BasisFunction object)
        :param orientation_angles: list of 3 Euler angles [pitch, yaw, roll]
        :param center: center of symmetry group [x, y, z]
        """

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
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])

    @property
    def self_similarity(self):
        return self._self_similarity


class SymmetrySingleDeterminant(SymmetryMolecule):
    """
    get symmetry from single determinat wave function
    """
    def __init__(self, group, alpha_orbitals, beta_orbitals, orientation_angles=None, center=None):
        """

        :param group: symmetry group
        :param alpha_orbitals: list of alpha orbitals (BasisFunction objects)
        :param beta_orbitals: list of beta orbitals (BasisFunction objects)
        :param orientation_angles: list of 3 Euler angles [pitch, yaw, roll]
        :param center: center of symmetry group [x, y, z]
        """

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


class SymmetryMultiDeterminant(SymmetryMolecule):
    """
    Get symmetry from multi determinant wave function
    """
    def __init__(self, group, orbitals, configurations, orientation_angles=None, center=None):
        """

        :param group: symmetry group
        :param orbitals: list BasisFunction objects
        :param configurations: dictionary that contains the electronic configuration (see README for example)
        :param orientation_angles: list of 3 Euler angles [pitch, yaw, roll]
        :param center: center of symmetry group [x, y, z]
        """
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

    from posym.algebra import norm

    coordinates = [[ 0.000000000+00,  0.000000000+00,  2.40297090e-01],
                   [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
                   [ 1.43261539e+00,  1.75444785e-16, -9.61188362e-01]]

    symbols = ['O', 'H', 'H']

    sm = SymmetryAtomDisplacements('c2v', coordinates, symbols)
    print(sm.get_point_group())
    print(sm)
    mb = SymmetryMolecule('c2v', coordinates, symbols)
    print('Coor measure: ', mb, '(', norm(mb), ')')
