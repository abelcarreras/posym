# Example script that analyzes the continuous symmetry of
# the SCF density along the distortion of the H2O molecule
# This example requires PyQchem to compute the electronic density

from pyqchem import get_output_from_qchem, Structure, QchemInput
from pyqchem.parsers.basic import basic_parser_qchem
from posym import PointGroup, SymmetryFunction
import matplotlib.pyplot as plt
import numpy as np
from posym.tools import build_density, get_basis_set
import posym.algebra as al

measures = []
energies = []
frequencies = []
point_group = 'c2v'

scan_range = np.arange(-0.2, 0.21, 0.01)
for x_dist in scan_range:

    water = [[x_dist,           0.00000000e+00,  2.40297090e-01],
             [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
             [1.43261539e+00,   1.75444785e-16, -9.61188362e-01]]
    water = np.array(water) * 0.529177249

    molecule_water = Structure(coordinates=water,
                               symbols=['O', 'H', 'H'],
                               charge=0,
                               multiplicity=1)

    qc_input = QchemInput(molecule_water,
                          jobtype='sp',
                          exchange='hf',
                          basis='sto-3g',
                          sym_ignore=True,
                          )

    parsed_data, ee = get_output_from_qchem(qc_input, parser=basic_parser_qchem, return_electronic_structure=True)

    molecule_coor = np.array(ee['structure'].get_coordinates())
    molecule_symbols = np.array(ee['structure'].get_symbols())
    density_matrix = ee['total_scf_density']

    basis_set = get_basis_set(molecule_coor, ee['basis'])

    f_density = build_density(basis_set, density_matrix)
    print('density integral: ', f_density.integrate)


    print('Final energy:', parsed_data['scf_energy'])
    energies.append(parsed_data['scf_energy'])

    sm = SymmetryFunction(point_group, f_density)

    measures.append(sm.get_ir_representation().values)

    print('density measure: ', sm, ' (', al.norm(sm), ')')
    print('position measure: ', sm.measure_pos)
    print('----------------------\n')

pg = PointGroup(group=point_group)

plt.title('density')
for ir, l in zip(np.array(measures).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.figure()
plt.title('energy')
plt.plot(scan_range, energies, '-')
plt.ylabel('Hartree')
plt.xlabel('Bohr')

plt.show()



