from posym import SymmetryBase, PointGroup
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def get_atom_map(basis):
    atom_map = []
    for i, atoms in enumerate(basis['atoms']):
        for shell in atoms['shells']:
            atom_map += [i] * shell['functions']

    return atom_map

def get_simplified_matrx(matrix, atom_map):

    counts = np.unique(atom_map, return_counts=True)[1]

    n_red = len(np.unique(atom_map))
    smatrix = np.zeros((n_red, n_red), dtype=float)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            smatrix[atom_map[i], atom_map[j]] += value  # / np.sqrt(counts[atom_map[i]] * counts[atom_map[j]])

    return smatrix


class SymmetryOperator(SymmetryBase):
    def __init__(self, group, coordinates, symbols, operator_matrix, basis, optimize=True):

        ir_table = PointGroup(group).ir_table

        # set coordinates at geometrical center
        self._coordinates = np.array([c - np.average(coordinates, axis=0) for c in coordinates])

        self._operator_matrix = operator_matrix
        self._basis = basis
        self._symbols = symbols

        self._coor_measures = []
        self._operator_measures = []

        atom_map = get_atom_map(self._basis)

        simp_matrix = get_simplified_matrx(self._operator_matrix, atom_map)

        if optimize:
            self._angles = self.get_orientation(ir_table.operations)
        else:
            self._angles = [0, 0, 0]

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        for operation in ir_table.operations:
            operations_dic = ir_table.get_all_operations()
            matrix_measures = []
            for op in operations_dic[operation.label]:
                matrix_m = op.get_measure_op(self._coordinates, self._symbols, simp_matrix, orientation=rotmol)
                matrix_measures.append(matrix_m)

            matrix_measures = np.average(matrix_measures)
            self._operator_measures.append(matrix_measures)


        total_state = pd.Series(self._operator_measures, index=ir_table.index)

        super().__init__(group, total_state)

    def get_orientation(self, operations):

        def optimization_function(angles):

            rotmol = R.from_euler('zyx', angles, degrees=True)

            coor_measures = []
            for operation in operations:
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

        return res.x

    @property
    def get_measure_pos(self):
        return np.product(self._coor_measures)

    @property
    def opt_coordinates(self):
        rotmol = R.from_euler('zyx', self._angles, degrees=True)
        return rotmol.apply(self._coordinates)

if __name__ == '__main__':

    from pyqchem import get_output_from_qchem, Structure, QchemInput
    from pyqchem.parsers.parser_optimization import basic_optimization
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

    #coordinates = [[ 0.0000000000,    0.0000000000,    0.0000000000],
    #               [ 0.5541000000,    0.7996000000,    0.4965000000],
    #               [ 0.6833000000,   -0.8134000000,   -0.2536000000],
    #               [-0.7782000000,   -0.3735000000,    0.6692000000],
    #               [-0.4593000000,    0.3874000000,   -0.9121000000]]

    coordinates_ = [[-3.11301739e-06,  1.12541091e-05, -7.97835696e-06],
                   [-6.58614327e-02, -7.77865103e-01, -7.63860353e-01],
                   [-3.64119514e-02,  9.82136003e-01, -4.76386979e-01],
                   [-8.37587195e-01, -9.93456888e-02,  6.93899368e-01],
                   [ 9.39879258e-01, -1.04992740e-01,  5.46395830e-01]]

    symbols_ = ['C', 'H', 'H', 'H', 'H']

    molecule = Structure(coordinates, symbols)
    molecule = get_geometry_from_pubchem('naphthalene')

    qc_input = QchemInput(molecule,
                          jobtype='opt',
                          exchange='b3lyp',
                          basis='sto-3g',
                          scf_print=3,
                          )

    parsed_data, ee = get_output_from_qchem(qc_input,
                                            # parser=basic_optimization,
                                            store_full_output=True,
                                            read_fchk=True, processors=6)

    molecule_coor = np.array(ee['structure'].get_coordinates())
    molecule_symbols = np.array(ee['structure'].get_symbols())

    matrix = ee['overlap']
    group = ee['structure'].get_point_symmetry()
    print('group:', group)

    so = SymmetryOperator(group, molecule_coor, molecule_symbols, matrix, ee['basis'], optimize=True)

    print(so)