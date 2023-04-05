# Example script that analyzes the continuous symmetry of
# the SCF density along the distortion of the H2O molecule
# This example requires PyQchem to compute the electronic density

from pyqchem import Structure
from posym import PointGroup, SymmetryMoleculeBase
import matplotlib.pyplot as plt
import numpy as np
import posym.algebra as al

measures = []
point_group = 'c2v'

scan_range = np.arange(-1, 1.2, 0.2)
for x_dist in scan_range:

    water = [[x_dist,           0.00000000e+00,  2.40297090e-01],
             [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
             [1.43261539e+00,   1.75444785e-16, -9.61188362e-01]]
    water = np.array(water) * 0.529177249 * 3

    molecule_water = Structure(coordinates=water,
                               symbols=['O', 'H', 'H'],
                               charge=0,
                               multiplicity=1)


    sm = SymmetryMoleculeBase(point_group, molecule_water.get_coordinates(), molecule_water.get_symbols())

    measures.append(sm.get_ir_representation().values)

    print('density measure: ', sm, ' (', al.norm(sm), ')', sm.get_ir_representation())
    print('position measure: ', sm.measure_pos)
    print('----------------------\n')


pg = PointGroup(group=point_group)

plt.title('coordinates')
for ir, l in zip(np.array(measures).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.show()



