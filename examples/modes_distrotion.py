# Example script that analyzes the continuous symmetry of
# the normal modes along the distortion of the H2O molecule
# This example requires PyQchem to compute the normal modes

from pyqchem import get_output_from_qchem, Structure, QchemInput
from pyqchem.parsers.parser_frequencies import basic_frequencies
from posym import PointGroup, SymmetryNormalModes
import matplotlib.pyplot as plt
import numpy as np


measures_m0 = []
measures_m1 = []
measures_m2 = []
energies = []
frequencies = []

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
                          jobtype='freq',
                          exchange='hf',
                          basis='6-31G',
                          sym_ignore=True,
                          )

    parsed_data, ee = get_output_from_qchem(qc_input, parser=basic_frequencies, return_electronic_structure=True)

    molecule_coor = np.array(ee['structure'].get_coordinates())
    molecule_symbols = np.array(ee['structure'].get_symbols())

    # print(' structure')
    print('Final energy:', parsed_data['scf_energy'])
    energies.append(parsed_data['scf_energy'])
    modes = [np.array(m['displacement']) for m in parsed_data['modes']]
    freqs = [m['frequency'] for m in parsed_data['modes']]
    print('freqs: ', freqs)
    frequencies.append(freqs)

    sm = SymmetryNormalModes(group='c2v', coordinates=molecule_coor, modes=modes, symbols=molecule_symbols)

    m = 1  # mode number
    measures_m0.append(sm.get_state_mode(0).get_ir_representation().values)
    measures_m1.append(sm.get_state_mode(1).get_ir_representation().values)
    measures_m2.append(sm.get_state_mode(2).get_ir_representation().values)

    print('sm', sm.get_state_mode(m))

    print(sm.measure_pos)
    for i in range(len(modes)):
        print('m {}:'.format(i + 1), sm.get_state_mode(i))
    print('----------------------\n')


pg = PointGroup(group='C2v')

plt.title('mode 1')
for ir, l in zip(np.array(measures_m0).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.figure()
plt.title('mode 2')
for ir, l in zip(np.array(measures_m1).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.figure()
plt.title('mode 3')
for ir, l in zip(np.array(measures_m2).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.figure()
plt.title('Frequencies')
for i, freq in enumerate(np.array(frequencies).T):
    plt.plot(scan_range, freq, '-', label='mode: {}'.format(i+1))
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylabel('Frequency (cm-1)')

plt.figure()
plt.title('energy')
plt.plot(scan_range, energies, '-')
plt.ylabel('Hartree')
plt.xlabel('Bohr')

plt.show()



